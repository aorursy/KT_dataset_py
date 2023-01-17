import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
data = pd.read_csv('../input/mall-customers/Mall_Customers.csv')



print(data.head())
data.describe()
data.isna().sum()
sns.pairplot(data[data.columns.drop('CustomerID')],hue='Genre')

plt.show()
sns.heatmap(data.corr(), annot=True)
male_customers = data[data['Genre'] == 'Male']

female_customers = data[data['Genre'] == 'Female']

sns.heatmap(female_customers.corr(), annot=True);

plt.title('Correlation Heatmap - Female');
sns.heatmap(male_customers.corr(), annot=True);

plt.title('Correlation Heatmap - Male');
data.drop(['Genre'], axis=1, inplace=True)

data.drop(['Age'], axis=1, inplace=True)

data.head()

X = data.iloc[:, [1, 2]].values
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS (Within Cluster Sum of Squares)')

plt.show()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)

pred = kmeans.fit_predict(X)
from sklearn import metrics

score1 = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean')

print("Score of KMeans = ", score1)
# Visualising the clusters

plt.figure(figsize=(15,10))

plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[pred == 4, 0], X[pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'black', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(X)

labels_agg = hc.labels_
score_agg = metrics.silhouette_score(data,labels_agg)



print("Score of Agglomerative =", score_agg)
plt.figure(figsize=(15,10))

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()