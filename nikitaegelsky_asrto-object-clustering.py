import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.cluster import KMeans, MeanShift, DBSCAN

from sklearn.mixture import GaussianMixture

from scipy.stats import multivariate_normal as mvn
dataset = pd.read_csv('../input/space-object-dimensions/experiments.csv')
dataset.head()
plt.figure(figsize=(14,7)) 

plt.scatter(dataset["x"],dataset["y"], color="purple", s=150)

plt.xlabel('x')

plt.ylabel('y')

plt.title('Data Distribution')

plt.show()
dataset=dataset.drop(dataset.columns[0], axis=1)
dataset.info()
X = dataset.iloc[:,[0,1]].values
dist=[]

lst = [1,2,3,4,5,6,7,8,9,10,11]

for i in lst:

  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=61)

  kmeans.fit(X)

  dist.append(kmeans.inertia_)
plt.figure(figsize=(10,6)) 

plt.plot(lst, dist, color='r', lw=2)

plt.title('Elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('Distance between clusters')

plt.show()
kmeansmodel = KMeans(n_clusters=4, init='k-means++', random_state=0)

y_kmeans= kmeansmodel.fit_predict(X)

np.unique(y_kmeans)
plt.figure(figsize=(17,7))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 200, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 200, c = 'gray', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 200, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 200, c = 'cyan', label = 'Cluster 4')

plt.title('K_means')

plt.xlabel('X')

plt.ylabel('Y')

plt.legend(loc='upper left',prop={'size': 16})

plt.show()
k_means_score=round(metrics.silhouette_score(X, y_kmeans),4)*100

print("Silhouette score =",k_means_score,"%")
meanShift = MeanShift(bandwidth=2).fit(X)

print("Objects have been divided in", len(np.unique(meanShift.labels_)),"clusters.")
plt.figure(figsize=(17,7))

plt.scatter(X[meanShift.labels_ == 0, 0], X[meanShift.labels_ == 0, 1], s = 200, c = 'red', label = 'Cluster 1')

plt.scatter(X[meanShift.labels_ == 1, 0], X[meanShift.labels_ == 1, 1], s = 200, c = 'gray', label = 'Cluster 2')

plt.scatter(X[meanShift.labels_ == 2, 0], X[meanShift.labels_ == 2, 1], s = 200, c = 'green', label = 'Cluster 3')

plt.scatter(X[meanShift.labels_ == 3, 0], X[meanShift.labels_ == 3, 1], s = 200, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[meanShift.labels_ == 4, 0], X[meanShift.labels_ == 4, 1], s = 200, c = 'violet', label = 'Cluster 5')

plt.title('MeanShift')

plt.xlabel('X')

plt.ylabel('Y')

plt.legend(loc='upper left',prop={'size': 16})

plt.show()
meanShift_score=round(metrics.silhouette_score(X, meanShift.labels_),4)*100

print("Silhouette score =",meanShift_score,"%")
DBScan = DBSCAN(eps=1.9, min_samples=5).fit(X)

print("Objects have been divided in", len(np.unique(DBScan.labels_)),"clusters.")
plt.figure(figsize=(17,7))

plt.scatter(X[DBScan.labels_ == 0, 0], X[DBScan.labels_ == 0, 1], s = 200, c = 'red', label = 'Cluster 1')

plt.scatter(X[DBScan.labels_ == 1, 0], X[DBScan.labels_ == 1, 1], s = 200, c = 'gray', label = 'Cluster 2')

plt.scatter(X[DBScan.labels_ == 2, 0], X[DBScan.labels_ == 2, 1], s = 200, c = 'green', label = 'Cluster 3')

plt.scatter(X[DBScan.labels_ == 3, 0], X[DBScan.labels_ == 3, 1], s = 200, c = 'cyan', label = 'Cluster 4')

plt.title('DBScan')

plt.xlabel('X')

plt.ylabel('Y')

plt.legend(loc='upper left',prop={'size': 16})

plt.show()
DBScan_score=round(metrics.silhouette_score(X, DBScan.labels_),4)*100

print("Silhouette score =",DBScan_score,"%")
gmm = GaussianMixture(n_components=4, covariance_type='full', init_params='random', random_state=5).fit(X)

gmm_labels = gmm.predict(X)

probs = gmm.predict_proba(X)

np.unique(gmm_labels)
plt.figure(figsize=(17,7))

plt.scatter(X[gmm_labels == 0, 0], X[gmm_labels == 0, 1], s = 200, c = 'red', label = 'Cluster 1')

plt.scatter(X[gmm_labels == 1, 0], X[gmm_labels == 1, 1], s = 200, c = 'gray', label = 'Cluster 2')

plt.scatter(X[gmm_labels == 2, 0], X[gmm_labels == 2, 1], s = 200, c = 'green', label = 'Cluster 3')

plt.scatter(X[gmm_labels == 3, 0], X[gmm_labels == 3, 1], s = 200, c = 'cyan', label = 'Cluster 4')

plt.title('Gaussian Mixture')

plt.xlabel('X')

plt.ylabel('Y')

plt.legend(loc='upper left',prop={'size': 16})

plt.show()
gmm_score=round(metrics.silhouette_score(X, gmm_labels),4)*100

print("Silhouette score =", gmm_score,"%")
d = {'Method': ["k-means", "mean shift", "DBSCAN", "GMM"], 'Score': [k_means_score, meanShift_score, DBScan_score, gmm_score]}

df = pd.DataFrame(data=d)
print(df)