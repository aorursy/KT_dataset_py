import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (12,8)

sns.set_style('whitegrid')
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=1000, n_features=2, centers=5, cluster_std=1.7, random_state=101) 
data[0].shape
data[0]
data[1].shape
data[1]
plt.scatter(x=data[0][:,0], y=data[0][:,1], c=data[1], cmap='rainbow')
from sklearn.cluster import KMeans
# Kmeans need to know the clusters before hand, As we created the data; we know there are 5 clusters

kmeans = KMeans(n_clusters=5)
kmeans.fit(data[0])
centers = kmeans.cluster_centers_

centers
plt.scatter(centers[:,0], centers[:,1], s=200, c='black')
predicted = kmeans.labels_

predicted 
fig, axes = plt.subplots(1,2, sharey=True)

ax1 = axes[0]

ax2 = axes[1]



ax1.set_title('Original')

ax1.scatter(data[0][:,0], data[0][:,1], c=data[1],cmap='rainbow')



ax2.set_title('Kmeans')

ax2.scatter(data[0][:,0], data[0][:,1], c=predicted,cmap='rainbow')



# Ignore the color of plot, Instead focus on how well the model seperated the clusters.