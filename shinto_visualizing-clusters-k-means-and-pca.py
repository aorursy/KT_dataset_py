import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
iris = pd.read_csv("../input/Iris.csv")

X = iris.iloc[:, 1:5].values

y = pd.Categorical(iris['Species']).codes
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(X)

pca_2d = pca.transform(X)
import pylab as pl

for i in range(0, pca_2d.shape[0]):

    if y[i] == 0:

        c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')

    elif y[i] == 1:

        c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')

    elif y[i] == 2:

        c3 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='b', marker='*')

pl.legend([c1, c2, c3], ['Setosa', 'Versicolor', 'Virginica'])

pl.title('Iris dataset with 3 clusters and known outcomes')

pl.show()
pl.scatter(pca_2d[:,0], pca_2d[:, 1], c='black')

pl.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=111)

kmeans.fit(X)

for i in range(0, pca_2d.shape[0]):

    if kmeans.labels_[i] == 1:

        c1 = pl.scatter(pca_2d[i,0], pca_2d[i, 1], c='r', marker='+')

    elif kmeans.labels_[i] == 0:

        c2 = pl.scatter(pca_2d[i,0], pca_2d[i, 1], c='g', marker='o')

    elif kmeans.labels_[i] == 2:

        c3 = pl.scatter(pca_2d[i,0], pca_2d[i, 1], c='b', marker='*')

pl.legend([c1, c2, c3], ['Cluster 1', 'Cluster 0', 'Cluster 2'])

pl.title('K-means clusters the Iris dataset into 3 clusters')

pl.show()