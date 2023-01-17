import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.cluster import KMeans
dataset = pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')

dataset.head()
X = dataset.iloc[:,[3,4]].values

X
kmeans = KMeans(n_clusters=5, init='k-means++',random_state=42)

y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c='red', label='Cluster 1')

plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c='blue', label='Cluster 2')

plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c='green', label='Cluster 3')

plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c='cyan', label='Cluster 4')

plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c='magenta', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income(k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Careless')

plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Standard')

plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Target')

plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Sensible')

plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Careful')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
dataset["clusters"] = kmeans.labels_

dataset.sample(5)
centers = pd.DataFrame(kmeans.cluster_centers_)

centers
centers["clusters"] = range(5) # number of clusters

centers
dataset["ind"] = dataset.index

dataset.head()
dataset = dataset.merge(centers)

dataset
dataset.sample(20)
dataset = dataset.sort_values("ind")

dataset.head()
dataset = dataset.drop("ind",1)

dataset.head()
dataset = pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')

d = dataset.iloc[:,[2,3,4]]

X = dataset.iloc[:,[2,3,4]].values

X
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)

y_kmeans = kmeans.fit_predict(X)
dataset["clusters"] = kmeans.labels_
centers = pd.DataFrame(kmeans.cluster_centers_)

centers
centers["clusters"] = range(5) # number of clusters

dataset["ind"] = dataset.index

dataset = dataset.merge(centers)

dataset.sample(5)
dataset = dataset.sort_values("ind")

dataset = dataset.drop("ind",1)

dataset.head()
wcss = []

for i in range(1,11):

    km = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)

    km.fit(X)

    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('WCSS')

plt.show()