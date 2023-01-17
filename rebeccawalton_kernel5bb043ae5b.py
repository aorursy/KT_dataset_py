import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import operator

import statsmodels.api as sm

from IPython.core.display import display

import scipy.stats as ss

from scipy.stats import binom

from matplotlib.pyplot import plot, scatter, figure

from sklearn.cluster import KMeans

from statistics import mean 
# (1)

# Generate 'random' data from 3 clusters - first we'll make the centers far enough from each 

# other and the variance low enough that they're pretty easy to identify. 

# We'll also set the covariance such that our clusters are circular and equally sized for simplicity



cluster_1 = np.random.multivariate_normal([0.5, 0.5], [[1, 0],[0, 1]], 200) 

cluster_2 = np.random.multivariate_normal([7.5, 2.5], [[1, 0],[0, 1]], 200) 

cluster_3 = np.random.multivariate_normal([6, 7], [[1, 0],[0, 1]], 200)



plot(cluster_1[:,0], cluster_1[:,1], 'ro', cluster_2[:,0], cluster_2[:,1], 'bo', cluster_3[:,0], cluster_3[:,1], 'go', markersize=1)
# Let's see how well k-means does identifying these clusters

all_data = np.concatenate((cluster_1, cluster_2, cluster_3), axis=0)

df = pd.DataFrame({

    'x': all_data[:,0],

    'y': all_data[:,1]

})



kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

labels = kmeans.predict(df)



fig = figure(figsize=(5, 5))



colmap = {1: 'r', 2: 'g', 3: 'b'}

colors = [colmap[x+1] for x in labels]



scatter(df['x'], df['y'], color=colors, s=1) 
# Looks like it did pretty well. 

# What proportion did it successfully predict?



# The first 100 should have the same label - but we don't know which label as this is non-deterministic

cluster_1_label = round(mean([val.item() for val in labels[0:200]]))

print(cluster_1_label)

cluster_2_label = round(mean([val.item() for val in labels[200:400]]))

print(cluster_2_label)

cluster_3_label = round(mean([val.item() for val in labels[400:600]]))

print(cluster_3_label)

# These should all be different!



number_correctly_identified_as_1 = sum(l == cluster_1_label for l in labels[0:200])

number_correctly_identified_as_2 = sum(l == cluster_2_label for l in labels[200:400])

number_correctly_identified_as_3 = sum(l == cluster_3_label for l in labels[400:600])



accuracy = (number_correctly_identified_as_1 + number_correctly_identified_as_2 + number_correctly_identified_as_3)/600

print(accuracy)
# What did it think the cluster means were?

print(kmeans.cluster_centers_)

# Pretty accurate (on mine!)
# (2)

# We assumed our data had roughly equally sized, circular clusters. What happens when we break some of these assumptions?



cluster_1 = np.random.multivariate_normal([1, 1], [[1, 0],[0, 1]], 200) 

cluster_2 = np.random.multivariate_normal([9, 7], [[0.2, 0],[0, 0.2]], 200) 

cluster_3 = np.random.multivariate_normal([6, 7], [[1.7, 1.6],[1.6, 1.7]], 200)



plot(cluster_1[:,0], cluster_1[:,1], 'ro', cluster_2[:,0], cluster_2[:,1], 'bo', cluster_3[:,0], cluster_3[:,1], 'go', markersize=1)
all_data = np.concatenate((cluster_1, cluster_2, cluster_3), axis=0)

df = pd.DataFrame({

    'x': all_data[:,0],

    'y': all_data[:,1]

})



kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

labels = kmeans.predict(df)



fig = plt.figure(figsize=(5, 5))



colmap = {1: 'r', 2: 'g', 3: 'b'}

colors = [colmap[x+1] for x in labels]



scatter(df['x'], df['y'], color=colors, s=1) 
# Even though the clusters were still fairly obvious to the human eye, Because K-means assumed the clusters were round and equally sized

# it doesn't do very well. The points at (8,10) are closer to the center of the right-most cluster than the center of the long thin cluster

# so K-means will allocate them to the right-most cluster without taking into account the fact that the shape of the long thin cluster is

# relavant as it stretches in this direction, and the other cluster is very small so is unlikely to stretch this far.
# (3)

# PCA

# Let's generate some random 2D data points where the two features (x and y) are very correlated

two_dimensional_data = np.random.multivariate_normal([6, 7], [[1.7, 1.6],[1.6, 1.7]], 200)

plot(two_dimensional_data[:,0], two_dimensional_data[:,1], 'ro', markersize=1)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principal_components = pca.fit_transform(two_dimensional_data)



# Just take the first component as this is the one that explains the most variance

one_dimensional_data = principal_components[:,0]



plot(one_dimensional_data, one_dimensional_data, 'ro', markersize=1)

# It looks like we pretty much preserved the character of the data whilst storing half as much data (plotting it like this doesn't really

# prove that, but it's difficult to super-impose a new axis over the existing plot and show which individual point on the original 

# corresponds to which point on the new one, so it's just meant as a visualisation!)