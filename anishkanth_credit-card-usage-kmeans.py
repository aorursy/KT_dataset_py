# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab as pl



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

card1 = pd.read_csv('../input/CreditCardUsage.csv')
card1.shape
card1.info()
card1.head()
card1.isna().sum()
card1[(card1['PAYMENTS']==0)].shape
card1[(card1['PAYMENTS']!=0) & (card1['MINIMUM_PAYMENTS'].isna())]
card1 = card1.drop('CUST_ID',axis = 1)
Hcard1 = card1
card1 = card1.dropna()
from sklearn.preprocessing import StandardScaler

card = StandardScaler().fit_transform(card1)
Hcard = card
from sklearn.cluster import KMeans

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(card)
k_means_labels = k_means.labels_

k_means_labels
k_means_cluster_centers = k_means.cluster_centers_

k_means_cluster_centers
import matplotlib.pyplot as plt 

# Initialize the plot with the specified dimensions.

fig = plt.figure(figsize=(10, 6))



# Colors uses a color map, which will produce an array of colors based on

# the number of labels there are. We use set(k_means_labels) to get the

# unique labels.

colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))



# Create a plot

ax = fig.add_subplot(1, 1, 1)



# For loop that plots the data points and centroids.

# k will range from 0-3, which will match the possible clusters that each

# data point is in.

for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):



    # Create a list of all data points, where the data poitns that are 

    # in the cluster (ex. cluster 0) are labeled as true, else they are

    # labeled as false.

    my_members = (k_means_labels == k)

    

    # Define the centroid, or cluster center.

    cluster_center = k_means_cluster_centers[k]

    

    # Plots the datapoints with color col.

    ax.plot(card[my_members, 0], card[my_members, 1], 'w', markerfacecolor=col, marker='.')

    

    # Plots the centroids with specified color, but with a darker outline

    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)



# Title of the plot

ax.set_title('KMeans')



# Remove x-axis ticks

ax.set_xticks(())



# Remove y-axis ticks

ax.set_yticks(())



# Show the plot

plt.show()



Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(card).score(card) for i in range(len(kmeans))]

score

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(card).inertia_ for i in range(len(kmeans))]

score

plt.figure(figsize=(10,6))

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Sum of within sum square')

pl.title('Elbow Curve')

pl.show()
clusterNum = 7

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(card)

labels = k_means.labels_

print(labels)
card1["Clus_km"] = labels
card1.head()
card1.groupby('Clus_km').mean()
import seaborn as sns
sns.scatterplot(data=card1,x='BALANCE',y='PURCHASES',hue = 'Clus_km')
sns.scatterplot(data=card1,x='PAYMENTS',y='PURCHASES',hue = 'Clus_km')
type(card1)
import scipy.cluster.hierarchy as sch



plt.figure(figsize=(15,10))

plt.title('Dendrogram')

plt.xlabel('BAL_PAY')

plt.ylabel('Euclidean distances')

plt.hlines(y=100000,xmin=0,xmax=100000,lw=3,linestyles='--')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(card1.iloc[:,[0,2]].values, method = 'ward'))

plt.show()