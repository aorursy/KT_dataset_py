%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

import numpy as np
from sklearn.datasets.samples_generator import make_blobs  #Make_blobs generates isotropic gaussian blobs for clustering

X, y_true = make_blobs(n_samples=300, centers=4,

                       cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50);
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')



centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);