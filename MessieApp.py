import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, silhouette_score, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering, DBSCAN

# Upload data
st.title("Εφαρμογή Ανάλυσης Δεδομένων")
uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(data.head())
    st.write("Data Info:")
    st.write(data.info())

    # Προεπεξεργασία δεδομένων
    label_col = data.columns[-1]
    st.write(f"Label Column: {label_col}")

    # Έλεγχος αν η ετικέτα είναι κατηγορική ή αριθμητική
    if data[label_col].dtype == 'object':
        st.write("Label is categorical")
        le = LabelEncoder()
        data[label_col] = le.fit_transform(data[label_col])
    else:
        st.write("Label is numeric")
        num_unique_labels = len(data[label_col].unique())
        if num_unique_labels > 20:
            st.write("Converting numeric label to categories using quartiles")
            data[label_col] = pd.qcut(data[label_col], q=4, labels=False)
        else:
            st.write("Converting numeric label to categories using LabelEncoder")
            data[label_col] = LabelEncoder().fit_transform(data[label_col])

    st.write("Processed Data:")
    st.write(data.head())
    st.write("Label unique values:", data[label_col].unique())

    # Select tab
    tabs = st.tabs(["2D Visualization", "Classification", "Clustering", "Info"])

    with tabs[0]:
        st.subheader("2D Visualization")
        method = st.selectbox("Επιλέξτε μέθοδο μείωσης διάστασης", ["PCA", "t-SNE"])
        
        if method == "PCA":
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(data.iloc[:, :-1])
            plt.figure(figsize=(16, 10))
            plt.scatter(pca_results[:, 0], pca_results[:, 1], c=data[label_col])
            plt.xlabel('pca-one')
            plt.ylabel('pca-two')
            plt.legend(data[label_col].unique())
            st.pyplot(plt)
        
        elif method == "t-SNE":
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
            tsne_results = tsne.fit_transform(data.iloc[:, :-1])
            plt.figure(figsize=(16, 10))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data[label_col])
            plt.xlabel('tsne-one')
            plt.ylabel('tsne-two')
            plt.legend(data[label_col].unique())
            st.pyplot(plt)

        # Exploratory Data Analysis (EDA)
        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("Histogram of each feature:")
        for column in data.columns[:-1]:
            plt.figure(figsize=(10, 4))
            plt.hist(data[column], bins=30)
            plt.title(f'Histogram of {column}')
            st.pyplot(plt)

        st.write("Pairplot of the data:")
        sns.pairplot(data, hue=label_col)
        st.pyplot(plt)

        # Heatmap for correlation (only numerical columns)
        st.write("Heatmap of correlation matrix:")
        numeric_data = data.select_dtypes(include=[float, int])
        fig, ax = plt.subplots()
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Classification")
        X = data.iloc[:, :-1]
        y = data[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        
        st.write("KNN Classification Report:")
        st.text(classification_report(y_test, y_pred_knn))
        st.write("KNN Accuracy Score:", accuracy_score(y_test, y_pred_knn))
        
        cm_knn = confusion_matrix(y_test, y_pred_knn)
        st.write("KNN Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        st.write("Random Forest Classification Report:")
        st.text(classification_report(y_test, y_pred_rf))
        st.write("Random Forest Accuracy Score:", accuracy_score(y_test, y_pred_rf))

        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        st.write("Logistic Regression Classification Report:")
        st.text(classification_report(y_test, y_pred_lr))
        st.write("Logistic Regression Accuracy Score:", accuracy_score(y_test, y_pred_lr))

        # Update comparison table
        results = pd.DataFrame({
            'Algorithm': ['KNN', 'Random Forest', 'Logistic Regression'],
            'Accuracy': [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)]
        })
        st.write(results)

    with tabs[2]:
        st.subheader("Clustering")
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        
        st.write("KMeans Silhouette Score:", silhouette_score(X, y_kmeans))
        
        st.write("KMeans Clustering Results:")
        st.write(pd.DataFrame({'Actual': y, 'Cluster': y_kmeans}).head(10))
        
        agg_clustering = AgglomerativeClustering(n_clusters=2)
        agg_clustering.fit(X)
        y_agg = agg_clustering.labels_

        st.write("Agglomerative Clustering Silhouette Score:", silhouette_score(X, y_agg))

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        y_dbscan = dbscan.fit_predict(X)

        st.write("DBSCAN Clustering Results:")
        st.write(pd.DataFrame({'Actual': y, 'Cluster': y_dbscan}).head(10))

        # Update comparison table
        clustering_results = pd.DataFrame({
            'Algorithm': ['KMeans', 'Agglomerative Clustering', 'DBSCAN'],
            'Silhouette Score': [silhouette_score(X, y_kmeans), silhouette_score(X, y_agg), silhouette_score(X, y_dbscan)]
        })
        st.write(clustering_results)
        
    with tabs[3]:
        st.subheader("Info")
        st.write("Αυτή η εφαρμογή αναπτύχθηκε από την ομάδα μας για το μάθημα Τεχνολογία Λογισμικού.")




        st.write("Η εφαρμογή προσφέρει εργαελεία ανάλυσης δεδομένων, επιτρέπει τη φόρτωση, την οπτικοποίηση, την κατηγοροιοποίηση και την ομαδοποίηση δεδομένων καθώς και την παρουσίαση των αποτελεσμάτων.")

        st.write("Λειτουργία της Εφαρμογής")

        st.write("1.Φόρτωση Δεδομένων: Επιλογή αρχείων CSV ή Excel")
        st.write("2.Οπτικοποίηση: 2D διαγράμματα με PCA ή t-SNE")
        st.write("3.EDA: Ιστογράμματα και pairplots για κατανόηση δεδομένων")
        st.write("4.Κατηγοριοποίηση: Αλγόριθμοι KNN και Random Forest")
        st.write("5.Ομαδοποίηση: Αλγόριθμοι K-Means και Agglomerative Clustering")
        

        st.write("Χρησιμοποιούμε τεχνολογίες Python, Streamlit, και scikit-learn.")
        st.write("Ομάδα Ανάπτυξης:")
        team_members = [
            {"name": "Θεοδόσης Καραγεωργίου", "am": "Inf2021076", "role": "Ανάπτυξη Κώδικα"},
            {"name": "Κωνσταντίνος Λιάβας", "am": "Inf2021121", "role": "Ανάλυση Δεδομένων"},
            {"name": "Φοίβος Ελευθερίου", "am": "Inf2021049", "role": "Οπτικοποίηση Δεδομένων"},
        ]
        for member in team_members:
            if "am" in member:
                st.write(f"{member['name']} (ΑΜ: {member['am']}): {member['role']}")
            else:
                st.write(f"{member['name']}: {member['role']}")


