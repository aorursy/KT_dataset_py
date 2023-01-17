import numpy as np

import pandas as pd



import matplotlib.pyplot as plt
data = np.random.rand(300,2)

data[:10]
plt.scatter(data[:,0], data[:,1])



plt.ylabel("Feature 1")

plt.xlabel("Feature 2")

plt.title("Data Scatter Plot")
from sklearn.cluster import KMeans

# 3 clusters

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
kmeans.cluster_centers_
kmeans.labels_
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))



ax1.scatter(data[:,0], data[:,1])

ax1.set_title("Original")

ax1.set_xlabel("Feature 1")

ax1.set_ylabel("Feature 2")



ax2.scatter(data[:,0], data[:,1], c=kmeans.labels_, cmap="autumn")

ax2.set_title("KMeans")

ax2.set_xlabel("Feature 1")

ax2.set_ylabel("Feature 2")
from sklearn.datasets import make_blobs
data_mb = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)

data_mb
data_mb[0]
plt.scatter(data_mb[0][:,0], data_mb[0][:,1], c=data_mb[1], cmap="autumn")
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(data_mb[0])
kmeans.cluster_centers_
kmeans.labels_
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))



ax1.scatter(data_mb[0][:,0], data_mb[0][:,1])

ax1.set_title("Original")

ax1.set_xlabel("Feature 1")

ax1.set_ylabel("Feature 2")



ax2.scatter(data_mb[0][:,0], data_mb[0][:,1], c=kmeans.labels_, cmap="autumn")

ax2.set_title("KMeans")

ax2.set_xlabel("Feature 1")

ax2.set_ylabel("Feature 2")