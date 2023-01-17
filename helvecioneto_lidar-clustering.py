!pip install laspy
import laspy

import scipy

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth

from sklearn import metrics

from sklearn import preprocessing

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import path



%matplotlib inline
# Open a file in read mode:

inFile = laspy.file.File("../input/_Merge_Remove Outliers_Normalize by DEM_Clip by Polygon.las")

# Grab a numpy dataset of our clustering dimensions:

dataset = np.vstack([inFile.x, inFile.y, inFile.z]).transpose()

print('Shape of dataset: ',dataset.shape)
print('Z range =', dataset[:, 2].max() - dataset[:, 2].min())

print('Z max =', dataset[:, 2].max(), 'Z min =', dataset[:, 2].min())

print('Y range =', dataset[:, 1].max() - dataset[:, 1].min())

print('Y max =', dataset[:, 1].max(), 'Y min =', dataset[:, 1].min())

print('X range =', dataset[:, 0].max() - dataset[:, 0].min())

print('X max =', dataset[:, 0].max(), 'X min =', dataset[:, 0].min())
plt.figure(figsize=(10,8))

ax = plt.axes(projection='3d')



# Data for a three-dimensional line

zline = dataset[:, 2]

xline = dataset[:, 0]

yline = dataset[:, 1]

ax.scatter3D(xline, yline, zline, 'gray', c=zline, cmap='Greens')

ax.view_init(90, 90)
Zlines = np.where(zline > 14, zline, np.nan)

Zline = np.where(zline > 14, zline, 0)

plt.figure(figsize=(10,8))

ax = plt.axes(projection='3d')



ax.scatter3D(xline, yline, Zline, 'gray', c=Zlines, cmap='Greens')

ax.view_init(0, 180)
new_dataset = np.vstack([inFile.x, inFile.y, Zline]).transpose()
#Clustering with DBSCAN

clustering = DBSCAN(eps=1, min_samples=10, leaf_size=1, metric='euclidean').fit(new_dataset)
labels = clustering.labels_

print(np.unique(labels))



plt.figure(figsize=(10,8))

ax = plt.axes(projection='3d')



ax.scatter3D(xline, yline, Zlines, 'gray', c=labels)

ax.view_init(0, 180)
bandwidth = estimate_bandwidth(new_dataset, quantile=1, n_samples=None, random_state=0, n_jobs=None)

print(bandwidth)
# ms = MeanShift(bandwidth=100, bin_seeding=None, cluster_all=True, min_bin_freq=1,

#     n_jobs=-1, seeds=None)



ms = MeanShift(bandwidth=10)



ms.fit(new_dataset)

labels = ms.labels_

cluster_centers = ms.cluster_centers_

n_clusters_ = len(np.unique(labels))
plt.figure(figsize=(10,8))

ax = plt.axes(projection='3d')



ax.scatter3D(xline, yline, Zlines, 'gray', c=labels)

ax.view_init(0, 180)