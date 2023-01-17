## MOST EASIEST WAY OF SEGMENTING CUSTOMERS

##DINESH SAGAR
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

data.head()
x=data.iloc[:,[3,4]].values

x
# lets build dendrogram. this is not mandatory,depends on us and our time lol.

import scipy.cluster.hierarchy as sch

plt.figure(figsize=(24, 8))

dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))

plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()
# LETS FIND IDEAL NUMBER OF CLUSTERS BY ELBOW METHOD



from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

    

plt.figure(figsize=(10, 8))

plt.plot(range(1, 11), wcss, marker='o')

plt.rcParams.update({'font.size': 18})

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#ideal number of clusters are 4, but i took 7

kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(x)
# Written so minimilastically without advance methods. any one can easily understand

# Created normal visualizations without much VFX (Just kidding :) !) Clean and neat plottings.



plt.figure(figsize=(24, 9))





plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')

plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')

plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')

plt.scatter(x[y_kmeans == 5, 0], x[y_kmeans == 5, 1], s = 50, c = 'k', label = 'Cluster 6')

plt.scatter(x[y_kmeans == 6, 0], x[y_kmeans == 6, 1], s = 50, c = 'brown', label = 'Cluster 7')





plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()



cluster1=('medium salary and medium spender')

cluster2=('medium salary and heavy spenders')

cluster3=('Low salary and medium spenders')

cluster4=('Low salary and low Spenders')

cluster5=('Low salary and heavy spenders')

cluster6=('High salary and low spenders')

cluster7=('High salary and high spenders')



print("The Category of people in Cluster 1 are: {}".format(cluster1))

print("The Category of people in Cluster 2 are: {}".format(cluster2))

print("The Category of people in Cluster 3 are: {}".format(cluster3))

print("The Category of people in Cluster 4 are: {}".format(cluster4))

print("The Category of people in Cluster 5 are: {}".format(cluster5))

print("The Category of people in Cluster 6 are: {}".format(cluster6))

print("The Category of people in Cluster 7 are: {}".format(cluster7))



print("BY THIS SIMPLE METHOD WE CAN CLUSTER CUSTOMERS AND SELL PRODUCTS BASED ON THIS")
