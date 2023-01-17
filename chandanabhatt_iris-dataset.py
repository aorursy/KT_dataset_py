import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.datasets import load_iris
plt.rcParams['figure.figsize'] = [10, 7]

plt.style.use('fivethirtyeight')
ds=load_iris() #loading the data from the iris dataset to ds, only the independent variables

data = pd.DataFrame(ds['data'], columns = ds['feature_names']) #converting the imported data into a dataframe
data.head(3)
data.shape
#there are 150 rows, 4 columns in the dataset
data.describe()
data.info()
#there are no null values in the dataset
x=data.iloc[:,:].values
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    

    kmeans = KMeans(n_clusters = i, random_state = 0, init = 'k-means++')

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1,11), wcss)

plt.title('Elbow Method')

plt.xlabel('K Clusters')

plt.ylabel('WCSS (Error)')

plt.show()
# Generating Model



kmeans = KMeans(n_clusters = 3, random_state = 0, init = 'k-means++')

y_kmeans = kmeans.fit_predict(x)
# Visualization



plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1],s=100, c = 'c', label = 'Cluster1')

plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1],s=100, c = 'g', label = 'Cluster2')

plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1],s=100, c = 'y', label = 'Cluster3')



plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 300, 

            c = 'r', alpha=0.6,label = 'Centroids')

plt.legend()

plt.show()
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method = 'single'),p=5,truncate_mode='level')

plt.title('Dendrogram')

plt.ylabel('Euclidean Distance')

plt.show()
# considering the clusters as 3 optimal number of clusters
k=3
# Generating Model



from sklearn.cluster import AgglomerativeClustering

agglo = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')

y_agglo = agglo.fit_predict(x)

plt.scatter(x[:,0], x[:,1], c = y_agglo)

# plt.show()

print(y_agglo)

print(y_agglo ==0)
plt.scatter(x[y_agglo==0,0], x[y_agglo==0,1], c = 'c', label = 'Cluster1')

plt.scatter(x[y_agglo==1,0], x[y_agglo==1,1], c = 'm', label = 'Cluster2')

plt.scatter(x[y_agglo==2,0], x[y_agglo==2,1], c = 'r', label = 'Cluster3')



plt.legend()

plt.show()
agglo = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

y_agglo = agglo.fit_predict(x)

plt.scatter(x[:,0], x[:,1], c = y_agglo)

# plt.show()

print(y_agglo)

print(y_agglo ==0)
plt.scatter(x[y_agglo==0,0], x[y_agglo==0,1], c = 'c', label = 'Cluster1')

plt.scatter(x[y_agglo==1,0], x[y_agglo==1,1], c = 'm', label = 'Cluster2')

plt.scatter(x[y_agglo==2,0], x[y_agglo==2,1], c = 'r', label = 'Cluster3')



plt.legend()

plt.show()
agglo = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'single')

y_agglo = agglo.fit_predict(x)

plt.scatter(x[:,0], x[:,1], c = y_agglo)

# plt.show()

print(y_agglo)

print(y_agglo ==0)
plt.scatter(x[y_agglo==0,0], x[y_agglo==0,1], c = 'c', label = 'Cluster1')

plt.scatter(x[y_agglo==1,0], x[y_agglo==1,1], c = 'm', label = 'Cluster2')

plt.scatter(x[y_agglo==2,0], x[y_agglo==2,1], c = 'r', label = 'Cluster3')



plt.legend()

plt.show()
agglo = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'average')

y_agglo = agglo.fit_predict(x)

plt.scatter(x[:,0], x[:,1], c = y_agglo)

# plt.show()

print(y_agglo)

print(y_agglo ==0)
plt.scatter(x[y_agglo==0,0], x[y_agglo==0,1], c = 'c', label = 'Cluster1')

plt.scatter(x[y_agglo==1,0], x[y_agglo==1,1], c = 'm', label = 'Cluster2')

plt.scatter(x[y_agglo==2,0], x[y_agglo==2,1], c = 'r', label = 'Cluster3')



plt.legend()

plt.show()