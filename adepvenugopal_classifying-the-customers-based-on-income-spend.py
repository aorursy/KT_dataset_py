##Importing the packages

#Data processing packages

import numpy as np 

import pandas as pd 



#Visualization packages

import matplotlib.pyplot as plt 

import seaborn as sns 



from sklearn.cluster import KMeans

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering



import warnings

warnings.filterwarnings('ignore')
#Import Mall Customer data

data = pd.read_csv('../input/Mall_Customers.csv')
#Find the size of the data Rows x Columns

data.shape
#Display first 5 rows of the data

data.head()
#Find Basic Statistics like count, mean, standard deviation, min, max etc.

data.describe()
#Find the the information about the fields, field datatypes and Null values

data.info()
#Extract Annual Income (k$) and Spending Score (1-100) fields 

target = data.iloc[:,[3,4]]
#Convert to Dataframe to  numpy array

X = np.array(target)

X
plt.figure(figsize=(24,12))

#plt.scatter(X[:,0], X[:,1], s = 25)

plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], s = 25)

plt.title('Raw Data',fontsize=25)

plt.xlabel('Annual Income (k$)',fontsize=25)

plt.ylabel('Spending Score (1-100)',fontsize=25)

plt.show()
#Elbow Method' helps to determine the appropriate number of clusters to use

wcss = [] #Within-Cluster Sum of Square (WCSS)

iterations = 500

num_centroid_seeds = 10

rand_state = 0



for i in range(1,11):

    kmeans = KMeans(n_clusters = i, max_iter = iterations, n_init = num_centroid_seeds, random_state = rand_state)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1,11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('WCSS')

plt.show()
#Finding kmeans using no. of clusters = 5

kmeans = KMeans(n_clusters = 5, max_iter = iterations, n_init = num_centroid_seeds, random_state = rand_state)

kmeans_preds = kmeans.fit_predict(X)

kmeans_preds
point_size = 25

colors = ['red', 'blue', 'green', 'cyan', 'magenta']

labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']



plt.figure(figsize = (18,12))

for i in range(5):

    plt.scatter(X[kmeans_preds == i,0], X[kmeans_preds == i,1], s = point_size, c = colors[i], label = labels[i])

    

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'orange', label = 'Centroids')

plt.title('Clusters of Clients')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend(loc = 'best')

plt.show()
plt.figure(figsize = (25,10))

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))



plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean Distances')

plt.show()


agg_clustering = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

agg_preds = agg_clustering.fit_predict(X)
point_size = 25

colors = ['red', 'blue', 'green', 'cyan', 'magenta']

labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']



plt.figure(figsize = (18,12))

for i in range(5):

    plt.scatter(X[agg_preds == i,0], X[agg_preds == i,1], s = point_size, c = colors[i], label = labels[i])



plt.title('Clusters of Clients')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend(loc = 'best')

plt.show()
point_size = 25

colors = ['red', 'blue', 'green', 'cyan', 'magenta']

labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']



plt.figure(figsize = (25,7))



plt.subplot(1,2,1)

for i in range(5):

    plt.scatter(X[kmeans_preds == i,0], X[kmeans_preds == i,1], s = point_size, c = colors[i], label = labels[i])

    

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'orange', label = 'Centroids')

plt.title('Clusters of Clients (K-Means)')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend(loc = 'best')



plt.subplot(1,2,2)

for i in range(5):

    plt.scatter(X[agg_preds == i,0], X[agg_preds == i,1], s = point_size, c = colors[i], label = labels[i])

    

plt.title('Clusters of Clients (Agglomerative)')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend(loc = 'best')



plt.show()