import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/Sample_Stocks.csv')
dataset.head()
dataset.count()
dataset.max()
dataset.min()
dataset.shape
for i in dataset.columns:

    print(i,"---",dataset[i].dtype)
dataset.isna().sum()
X = dataset.iloc[:, [1,0]].values
X
from sklearn.cluster import KMeans
#elbow method to find optimal number of clusters

wcss = []

for i in range(1, 8):

    clust = KMeans(n_clusters = i, init='k-means++',n_init=10, max_iter=300,tol=0.0001,random_state=42)

    clust.fit(X)

    wcss.append(clust.inertia_)
plt.plot(range(1, 8), wcss)

plt.plot(range(1, 8), wcss,'bo')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS SCORES')

plt.show()
# Fitting K-Means to the dataset

clust2 = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)

y_clust2 = clust2.fit_predict(X)
# Visualising the clusters

plt.scatter(X[y_clust2 == 0, 0], X[y_clust2 == 0, 1], s = 10, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_clust2 == 1, 0], X[y_clust2 == 1, 1], s = 10, c = 'green', label = 'Cluster 2')

plt.scatter(X[y_clust2 == 2, 0], X[y_clust2 == 2, 1], s = 10, c = 'magenta', label = 'Cluster 3')

plt.xlim([-2,6])

plt.ylim([-30,60])

plt.title('Sample Stocks clusters')

plt.xlabel("Dividend's Yield")

plt.ylabel('Returns')

plt.legend()

plt.show()
# Visualising the clusters based on study

plt.scatter(X[y_clust2 == 0, 0], X[y_clust2 == 0, 1], s = 10, c = 'red', label = 'worst stocks')

plt.scatter(X[y_clust2 == 1, 0], X[y_clust2 == 1, 1], s = 10, c = 'green', label = 'best stocks')

plt.scatter(X[y_clust2 == 2, 0], X[y_clust2 == 2, 1], s = 10, c = 'magenta', label = 'good stocks')

plt.xlim([-2,6])

plt.ylim([-30,60])

plt.title('Sample Stocks clusters')

plt.xlabel("Dividend's Yield")

plt.ylabel('Returns')

plt.legend()

plt.show()