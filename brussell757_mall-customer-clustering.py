# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Mall_Customers.csv')

df.head()
X = np.array(df.iloc[:,[3,4]])
import matplotlib.pyplot as plt



plt.scatter(X[:,0], X[:,1], s = 25)

plt.title('Raw Data')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.show()
from sklearn.cluster import KMeans



wcss = []

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
kmeans = KMeans(n_clusters = 5, max_iter = iterations, n_init = num_centroid_seeds, random_state = rand_state)

kmeans_preds = kmeans.fit_predict(X)
point_size = 25

colors = ['red', 'blue', 'green', 'cyan', 'magenta']

labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']



plt.figure(figsize = (10,7))

for i in range(5):

    plt.scatter(X[kmeans_preds == i,0], X[kmeans_preds == i,1], s = point_size, c = colors[i], label = labels[i])

    

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'orange', label = 'Centroids')

plt.title('Clusters of Clients')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend(loc = 'best')

plt.show()
import scipy.cluster.hierarchy as sch



plt.figure(figsize = (25,10))

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))



plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean Distances')

plt.show()
from sklearn.cluster import AgglomerativeClustering



agg_clustering = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

agg_preds = agg_clustering.fit_predict(X)
point_size = 25

colors = ['red', 'blue', 'green', 'cyan', 'magenta']

labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']



plt.figure(figsize = (10,7))

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