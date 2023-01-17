import numpy as np

import pandas as pd

import numpy as np

from sklearn.datasets import load_iris
iris = load_iris()

iris_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],

                     columns= iris['feature_names'] + ['target'])

iris_data = iris_data.drop(['target'], axis=1)

iris_data.head()
from sklearn.cluster import KMeans
k_means = KMeans(3)

k_means.fit(iris_data)
k_means.labels_
k_means.cluster_centers_
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
pca_2D = PCA(2, svd_solver='full')

features = pca_2D.fit_transform(iris_data)
plt.scatter(features[:, 0], features[:, 1])
variances = []

clusters = []

for i in range(1, 10):

    mean_model = KMeans(i)

    mean_model.fit(iris_data)

    clusters.append(i)

    variances.append(mean_model.inertia_)
plt.scatter(clusters, variances)
class K_Means_Clustering:

    def __init__(self, clusters, train):

        self.train = train

        self.clusters = clusters

        self.centers = np.zeroes((self.clusters, self.train.shape[0]))

        

    def euclidean_distance(self, row1, row2):

        distance = np.sqrt(np.sum((row1-row2)**2))

        return distance

    

    def train(self):

        self.clusters = np.random.randint(0, )