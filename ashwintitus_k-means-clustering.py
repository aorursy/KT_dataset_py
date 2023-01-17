import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

%matplotlib inline
x = -2 * np.random.rand(100,2)

plt.scatter(x[:,0],x[:,1])

plt.show()

x1 = 1 + 2 * np.random.rand(50,2)

plt.scatter(x1[:,0],x1[:,1])

plt.show()

x[50:100, :] = x1

plt.scatter(x[:,0], x[:,1], s=50, c='b') # s - size of points in the plot

plt.show()
Kmean = KMeans(n_clusters=2)

Kmean.fit(x)
Kmean.cluster_centers_
plt.scatter(x[:,0], x[:,1], s=50, c='b')

centroids = Kmean.cluster_centers_

print(centroids)

plt.scatter(centroids[0][0], centroids[0][1], c='y', marker='s', s=200)

plt.scatter(centroids[1][0], centroids[1][1], c='r', marker='s', s=200)

plt.show()
Kmean.labels_
Kmean.predict([[2,2]])