import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



from sklearn.cluster import KMeans, DBSCAN, OPTICS

from sklearn.datasets import make_blobs

from kmodes.kmodes import KModes

from sklearn.preprocessing import StandardScaler
"""Gaussian blobs for clustering"""

X_kmeans, y = make_blobs(n_samples=500, random_state=250)

X_kmeans = pd.DataFrame(StandardScaler().fit_transform(X_kmeans))

'''Apply K-Means'''

kmean_clusters =  KMeans(n_clusters=3, random_state=250).fit_predict(X_kmeans)



plt.figure(figsize = (8,8))

plt.title('K-Means Clustering',fontsize= 20)

plt.xlabel('Feature 1', fontsize=18)

plt.ylabel('Feature 2', fontsize=18)

f = plt.scatter(X_kmeans[0],X_kmeans[1],c=kmean_clusters)
"""Random categorical data"""

categorical_data = pd.DataFrame(np.random.choice(54, size = (200, 2)))

'''Apply K-Modes'''

km = KModes(n_clusters=3)

kmode_clusters = km.fit_predict(categorical_data)

kmode_clusters
centers = [[1, 1], [-1, -1], [1, -1]]

X_dbscan, labels_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.4, random_state=2)

X_dbscan = pd.DataFrame(StandardScaler().fit_transform(X_dbscan))

'''Apply DBSCAN'''

db = DBSCAN(eps=0.3, min_samples=10).fit(X_dbscan)

labels = db.labels_



plt.figure(figsize=(8,8))

plt.title('DBSCAN',fontsize= 20)

plt.xlabel('Feature 1',fontsize= 18)

plt.ylabel('Feature 2',fontsize= 18)

fig = plt.scatter(X_dbscan[0], X_dbscan[1], c= labels)
centers = [[1, 1], [-1, -1], [1, -1]]

X_optics, labels_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.4,random_state=2)

X_optics = pd.DataFrame(StandardScaler().fit_transform(X_optics))

'''Apply OPTICS'''

optics = OPTICS(xi=.05, min_cluster_size=.05, min_samples=30).fit(X_optics)

labels_optics = optics.labels_



plt.figure(figsize=(8,8))

plt.title('OPTICS',fontsize= 20)

plt.xlabel('Feature 1',fontsize= 18)

plt.ylabel('Feature 2',fontsize= 18)

fig = plt.scatter(X_optics[0], X_optics[1], c= labels_optics)