%reset -f

import time                   # To time processes

import warnings               # To suppress warnings



import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulation

import matplotlib.pyplot as plt                   # For graphics



from sklearn import cluster, mixture              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset



import os                     # For os related operations

import sys                    # For data size



X= pd.read_csv("../input/2017.csv", header=0)

X.columns.values

X.shape

X = X.iloc[:, 2:]

# Ignore Country and Happiness_Rank columns

X.head(2)

X.dtypes

X.info

#1. Normalize dataset

ss = StandardScaler()

ss.fit_transform(X)



#2.1 Begin Clustering 

n_clusters = 2



#2.2 KMeans

km = cluster.KMeans(n_clusters =n_clusters )

km_result = km.fit_predict(X)

plt.subplot(4, 2, 1)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=km_result)



#2.3 Mean Shift

bandwidth = 0.1

ms = cluster.MeanShift(bandwidth=bandwidth)

ms_result = ms.fit_predict(X)

plt.subplot(4, 2, 2)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=ms_result)



#2.4 Mini Batch K-Means

two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)

two_means_result = two_means.fit_predict(X)

plt.subplot(4, 2, 3)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c= two_means_result)



#2.5 Spectral clustering

spectral = cluster.SpectralClustering(n_clusters=n_clusters)

sp_result= spectral.fit_predict(X)

plt.subplot(4, 2, 4)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=sp_result)



#3. DBSCAN



eps = 0.3

dbscan = cluster.DBSCAN(eps=eps)

db_result= dbscan.fit_predict(X)

plt.subplot(4, 2, 5)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5], c=db_result)



# 4. Affinity Propagation

damping = 0.9

preference = -200

affinity_propagation = cluster.AffinityPropagation(damping=damping, preference=preference)

affinity_propagation.fit(X)

ap_result = affinity_propagation .predict(X)

plt.subplot(4, 2, 6)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=ap_result)



#5. Birch

birch = cluster.Birch(n_clusters=n_clusters)

birch_result = birch.fit_predict(X)

plt.subplot(4, 2, 7)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=birch_result)



#6. Gaussian Mixture modeling

gmm = mixture.GaussianMixture( n_components=n_clusters, covariance_type='full')

gmm.fit(X)

gmm_result = gmm.predict(X)

plt.subplot(4, 2, 8)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=gmm_result)
