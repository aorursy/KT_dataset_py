# importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# reading the dataset

data=pd.read_csv("../input/Mall_Customers.csv")
# displaying first 5 rows

data.head()
# (no. of rows, no. of columns)

data.shape
data.describe()
# finding any null values in data

data.isnull().sum()
# Finding the number of males and females in the data

data.Gender.value_counts()
# Visualising the number of males and females in the data

sns.countplot(x="Gender",data=data,palette="pastel")

plt.show()
# Computing minimum and maximum age of customers

print(min(data.Age))

print(max(data.Age))
# Visualising the age distribution of customers

plt.figure(figsize=(25,8))

sns.countplot(x="Age",data=data,palette="pastel")

plt.show()
# Visualising the age distribution of customers

plt.figure(figsize=(25,8))

sns.countplot(x="Annual Income (k$)",data=data,palette="pastel")

plt.show()
# Visualising the spending score of customers

plt.figure(figsize=(25,8))

sns.countplot(x="Spending Score (1-100)",data=data,palette="pastel")

plt.show()
# Finding correaltion between all the parameters in the dataset.

fig,ax = plt.subplots(figsize=(11,8))

sns.heatmap(data.corr(),annot=True,cmap="Blues" ,ax=ax)

plt.show()
# Taking annual income and spending score in x to make clusters

x=data.iloc[:,[3,4]]
# dispalying first 5 rows of x

x.head()
# Using elbow method to find the optimal number of clusters.

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):

    kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)
# Visualising elbow method

plt.plot(range(1,11),wcss)

plt.xlabel("No. of Clusters")

plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()
# creating object kmeans of class KMeans()

kmeans=KMeans(n_clusters=5,init="k-means++",max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)

y_kmeans

# it tells which data point belongs to which cluster (0,1,2,3,4)
y_kmeans.astype
# converting x into ndarray to avoid errors

x=np.array(x)
x.astype
fig = plt.figure(figsize=(25, 10))

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
import scipy.cluster.hierarchy as sch
# Visualising the dendrogram

fig = plt.figure(figsize=(25, 10))

dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))

plt.title("Dendrogram")

plt.xlabel("Customers")

plt.ylabel("Eucledian distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering
# creating object hc of class AgglomerativeClustering()

hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
# it gives an array which tells as to which data point belongs to which cluster (0,1,2,3,4)

y_hc=hc.fit_predict(x)
y_hc
y_hc.astype
# converting x into ndarray to avoid errors

x=np.array(x)
x.astype
# Visualising the clusters

fig = plt.figure(figsize=(25, 10))

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()