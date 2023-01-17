##Objective

#Learn various types of clustering algorithms as available in sklearn, We will use "World Happiness Report data" as dataset for clustering algorithms.

## 1.0 Call libraries

import numpy as np # linear algebra & data manipulation

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time                   # To time processes

import warnings               # To suppress warnings

import matplotlib.pyplot as plt                   # For graphics

from sklearn import cluster, mixture              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset

import os                     # For os related operations

import sys                    # For data size
# 2. Read data

X = pd.read_csv("../input/2017.csv", header=0)



# 3. Explore and scale

X.columns.values

X.shape                 # 155 X 12

X = X.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns

X.head(2)

X.dtypes

X.info



# 3.1 Normalize dataset for easier parameter selection

#    Standardize features by removing the mean and scaling to unit variance

# 3.1.2 Instantiate scaler object

ss = StandardScaler()

# 3.1.3 Use ot now to 'fit' &  'transform'

ss.fit_transform(X)

#### 4. Begin Clustering  (Methods that will be used for clustering analysis)  

#KMeans

#Mean Shift

#Mini Batch K-Means

#Spectral clustering   

#DBSCAN

#Affinity Propagation

#Birch

#Gaussian Mixture modeling



# 5.1 How many clusters

#     NOT all algorithms require this parameter

n_clusters = 2 
## 5 KMeans

                                 

# KMeans algorithm clusters data by trying to separate samples in n groups

#  of equal variance, minimizing a criterion known as the within-cluster sum-of-squares.                         



# 5.1 Instantiate object

km = cluster.KMeans(n_clusters =n_clusters )



# 5.2.1 Fit the object to perform clustering

km_result = km.fit_predict(X)



# 5.3 Draw scatter plot of two features, coloyued by clusters

plt.subplot(4, 2, 1)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=km_result)

plt.title("K-Means")
## 6. Mean Shift

# This clustering aims to discover blobs in a smooth density of samples.

#   It is a centroid based algorithm, which works by updating candidates

#    for centroids to be the mean of the points within a given region.

#     These candidates are then filtered in a post-processing stage to

#      eliminate near-duplicates to form the final set of centroids.

# Parameter: bandwidth dictates size of the region to search through. 



# 6.1

bandwidth = 0.1  



# 6.2 No of clusters are NOT predecided

ms = cluster.MeanShift(bandwidth=bandwidth)



# 6.3

ms_result = ms.fit_predict(X)



# 6.4

plt.subplot(4, 2, 2)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=ms_result)

plt.title("Mean Shift")
## 7. Mini Batch K-Means

#  Similar to kmeans but clustering is done in batches to reduce computation time



# 7.1 

two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)



# 7.2

two_means_result = two_means.fit_predict(X)



# 7.3

plt.subplot(4, 2, 3)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c= two_means_result)

plt.title("Mini Batch K-Means")

## 8. Spectral clustering   

# SpectralClustering does a low-dimension embedding of the affinity matrix

#  between samples, followed by a KMeans in the low dimensional space. It

#   is especially efficient if the affinity matrix is sparse.

#   SpectralClustering requires the number of clusters to be specified.

#     It works well for a small number of clusters but is not advised when 

#      using many clusters.



# 8.1

spectral = cluster.SpectralClustering(n_clusters=n_clusters)



# 8.2

sp_result= spectral.fit_predict(X)



# 8.3

plt.subplot(4, 2, 4)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=sp_result)

plt.title("Spectral clustering")
## 9. DBSCAN

#   The DBSCAN algorithm views clusters as areas of high density separated

#    by areas of low density. Due to this rather generic view, clusters found

#     by DBSCAN can be any shape, as opposed to k-means which assumes that

#      clusters are convex shaped.    

#    Parameter eps decides the incremental search area within which density

#     should be same



eps = 0.3



# 9.1 No of clusters are NOT predecided

dbscan = cluster.DBSCAN(eps=eps)



# 9.2

db_result= dbscan.fit_predict(X)



# 9.3

plt.subplot(4, 2, 5)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5], c=db_result)

plt.title("DBSCAN")
# 10. Affinity Propagation   

# Creates clusters by sending messages between pairs of samples until convergence.

#  A dataset is then described using a small number of exemplars, which are

#   identified as those most representative of other samples. The messages sent

#    between pairs represent the suitability for one sample to be the exemplar

#     of the other, which is updated in response to the values from other pairs. 

#       Two important parameters are the preference, which controls how many

#       exemplars are used, and the damping factor which damps the responsibility

#        and availability messages to avoid numerical oscillations when updating

#         these messages.



damping = 0.9

preference = -200



# 10.1  No of clusters are NOT predecided

affinity_propagation = cluster.AffinityPropagation(

        damping=damping, preference=preference)



# 10.2

affinity_propagation.fit(X)



# 10.3

ap_result = affinity_propagation .predict(X)



# 10.4

plt.subplot(4, 2, 6)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=ap_result)

plt.title("Affinity Propagation")
## 11. Birch

# The Birch builds a tree called the Characteristic Feature Tree (CFT) for the

#   given data and clustering is performed as per the nodes of the tree



# 11.1

birch = cluster.Birch(n_clusters=n_clusters)



# 11.2

birch_result = birch.fit_predict(X)



# 11.3

plt.subplot(4, 2, 7)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=birch_result)

plt.title("Birch")
# 12. Gaussian Mixture modeling

#  It treats each dense region as if produced by a gaussian process and then

#  goes about to find the parameters of the process



# 12.1

gmm = mixture.GaussianMixture( n_components=n_clusters, covariance_type='full')



# 12.2

gmm.fit(X)



# 12.3

gmm_result = gmm.predict(X)

plt.subplot(4, 2, 8)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=gmm_result)

plt.title("Gaussian Mixture modeling")

#########################################################