from sklearn import datasets

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN
iris = datasets.load_iris()

features = iris.data
scaler = StandardScaler()

features_std = scaler.fit_transform(features)
cluster = DBSCAN(n_jobs=-1)
model = cluster.fit(features_std)
model.labels_