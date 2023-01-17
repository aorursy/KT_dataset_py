import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
%matplotlib inline
np.random.seed(0)
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2,-1], [2,-3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')
k_means = KMeans(init='k-means++', n_clusters=4, n_init=12)
k_means = k_means.fit(X)
k_means_labels = k_means.labels_
k_means_labels
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers
# Initializing plot
fig = plt.figure(figsize=(6,4))

# based on number of labels color - Color will produce an array of colors.
# using set(k_means_labels) for getting unique labels
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# creating plot
ax = fig.add_subplot(1, 1, 1)

# For loop - plots data points and cemtroids.
# k will range from 0-3, which will match the possiible clusters that each data point is in.
for k, col in zip(range(len([[4,4], [-2,-1], [2,-3], [1, 1]])), colors):
    #Creating a list of all data points where the data points that are in the cluster are labeled as true
    # else they are labeled as false
    
    my_members = (k_means_labels == k)
    
    # Defining the centroids
    cluster_center = k_means_cluster_centers[k]
    
    # Plotting teh datapoints with the color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker = '.')
    
    # Plotting the centroids with the specified color, but with the darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    
# The title  of the plot
ax.set_title('KMeans')

# Remove x axis ticks
ax.set_xticks(())

# Remove y axis ticks
ax.set_yticks(())

# showing the plot

plt.show()
