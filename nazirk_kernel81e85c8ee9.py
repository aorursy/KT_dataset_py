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
df = pd.read_csv('../input/CreditCardUsage.csv')
df.shape
df.head()
df.drop(columns = 'CUST_ID', inplace = True)
df.head()
df.isna().sum()
df.CREDIT_LIMIT.nunique()
df[df.CREDIT_LIMIT.isna()== True]
df.CREDIT_LIMIT.mean()
df.iloc[5203,12] = 4500.0
df[df.MINIMUM_PAYMENTS.isna() == True]
df.MINIMUM_PAYMENTS.mean()
df.MINIMUM_PAYMENTS.fillna(865.0, inplace = True)
df.corr()
df.info()
from sklearn.preprocessing import StandardScaler

X = df.values[:,:]

X = np.nan_to_num(X)

clus_dataset = StandardScaler().fit_transform(X)

clus_dataset
from sklearn.cluster import KMeans

clusterNum = 3

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(X)

labels = k_means.labels_

print(labels)
import pylab as pl

Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

score

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()
#From the elbow curve arrived at optima K value to be 7 hence

clusterNum = 7

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(X)

labels = k_means.labels_

print(labels)
df.info()
df["clusters"] = labels
df.head(25)
import matplotlib.pyplot as plt 

plt.scatter(X[:, 0], X[:, 1], marker='.')
k_means_cluster_centers = k_means.cluster_centers_

%matplotlib inline

from sklearn.cluster import KMeans

k_means_labels = k_means.labels_

# Initialize the plot with the specified dimensions.

fig = plt.figure(figsize=(6, 4))



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

    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')

    

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
df.info()
df[['clusters','BALANCE','PURCHASES','CASH_ADVANCE_TRX','PAYMENTS']]
from   scipy.cluster.hierarchy import dendrogram, linkage

z   = linkage(df, method = 'median')



plt.figure(figsize=(20,7))



den = dendrogram(z)



plt.title('Dendrogram for the clustering of the dataset credit card usage)')

plt.xlabel('Type')

plt.ylabel('Euclidean distance in the space with other variables')