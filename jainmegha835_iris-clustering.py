data=pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
data.head(3)
data.columns
data.shape
data.info()
data.describe()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)
cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)
model = cluster.fit(features_std)

new_observation = [[0.8, 0.8, 0.8, 0.8]]
# Predict observation's cluster
model.predict(new_observation)
model.cluster_centers_
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
features, true_labels = make_moons(n_samples=250,noise=0.05,random_state=42)
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=2)
dbscan = DBSCAN(eps=0.3)
kmeans.fit(scaled_features)
dbscan.fit(scaled_features)
kmeans_silhouette = silhouette_score(
scaled_features, kmeans.labels_ ).round(2)
dbscan_silhouette = silhouette_score(
scaled_features, dbscan.labels_).round (2)
kmeans_silhouette
dbscan_silhouette