import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/Mall_Customers.csv')
dataset.head()
dataset.isnull().sum()
X = dataset.iloc[:, [3, 4]].values
from sklearn.cluster import KMeans

wcss = [] #Within Cluster Sum of Squares

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter=300, n_init=10, random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('No of the clusters')

plt.ylabel('WCSS')



plt.show()
kmeans = KMeans(n_clusters = 5, init='k-means++', max_iter=300, n_init=10, random_state=0)

y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans ==  0, 0], X[y_kmeans ==  0, 1], c = 'red', label = 'cluster1')

plt.scatter(X[y_kmeans ==  1, 0], X[y_kmeans ==  1, 1], c = 'blue', label = 'cluster2')

plt.scatter(X[y_kmeans ==  2, 0], X[y_kmeans ==  2, 1], c = 'green', label = 'cluster3')

plt.scatter(X[y_kmeans ==  3, 0], X[y_kmeans ==  3, 1], c = 'cyan', label = 'cluster4')

plt.scatter(X[y_kmeans ==  4, 0], X[y_kmeans ==  4, 1], c = 'magenta', label = 'cluster5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')

plt.title('Clusters of clients')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
plt.scatter(X[y_kmeans ==  0, 0], X[y_kmeans ==  0, 1], c = 'red', label = 'Careful')

plt.scatter(X[y_kmeans ==  1, 0], X[y_kmeans ==  1, 1], c = 'blue', label = 'Standard')

plt.scatter(X[y_kmeans ==  2, 0], X[y_kmeans ==  2, 1], c = 'green', label = 'Target')

plt.scatter(X[y_kmeans ==  3, 0], X[y_kmeans ==  3, 1], c = 'cyan', label = 'Careless')

plt.scatter(X[y_kmeans ==  4, 0], X[y_kmeans ==  4, 1], c = 'magenta', label = 'Sensible')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')

plt.title('Clusters of clients')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()