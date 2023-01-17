from sklearn.datasets.samples_generator import make_blobs

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
fig = plt.figure()

X, y = make_blobs(n_samples=1200, centers=4,

                  random_state=0, cluster_std=0.60)



plt.scatter(X[:, 0], X[:, 1], s=50)

plt.show()
from sklearn.cluster import KMeans

est = KMeans(4)  # 4 clusters

est.fit(X)

y_kmeans = est.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow')

plt.show()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))



# Obtain labels for each point in mesh. Use last trained model.

Z = est.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

Z = Z.reshape(xx.shape)



plt.clf()

plt.imshow(Z, interpolation='nearest',

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap=plt.cm.Paired,

           aspect='auto', origin='lower')



plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)

plt.show()