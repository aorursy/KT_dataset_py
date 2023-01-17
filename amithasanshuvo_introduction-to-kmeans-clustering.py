# Importing all the important libraries that we need



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn import datasets
# Loading the dataset from sklearn datasets



df = pd.read_csv('../input/iris/Iris.csv')

#y = iris.target

x = df.drop(['Id', 'Species'], axis = 1) 
# A glance of the dataset



#x
# Defining the number of clusters an



kmeans5 = KMeans(n_clusters=5)



#Output of Kmeans clustering with value 5

y_kmeans5 = kmeans5.fit_predict(x)

print(y_kmeans5)



#Print the centers of 5 clusters

kmeans5.cluster_centers_
# Printing the interia value



#  Inertia actually calculates the sum of distances of all the points within

#a cluster from the centroid of that cluster. It tells us how far the points 

#within a cluster are. The distance between them should be as low as possible.





kmeans5.inertia_
SSE =[]

for clusters in range(1, 11):

    kmeans = KMeans(n_clusters = clusters).fit(x)

    kmeans.fit(x)

    SSE.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1, 11), SSE)

plt.title('Elbow method')

plt.xlabel('No of clusters')

plt.ylabel('Inertia')

plt.show()
kmeans3 = KMeans(n_clusters = 3)
y_kmeans3 = kmeans3.fit_predict(x)

print(y_kmeans3)
# Printing the center points

kmeans3.cluster_centers_
# Let's see how many data points are in these 3 clusters.



frame = pd.DataFrame(x)

frame['cluster'] = y_kmeans3

frame['cluster'].value_counts()
plt.scatter(x.iloc[:,0],x.iloc[:,1], c = y_kmeans3, cmap='rainbow')