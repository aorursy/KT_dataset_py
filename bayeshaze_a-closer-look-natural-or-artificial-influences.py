# General imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

from sklearn import decomposition

%matplotlib inline
# Read the data file.

df = pd.read_csv("../input/HR_comma_sep.csv")

# Looking here only at the employees that left

df2 = df[df['left'] == 1].copy()

df2.reset_index(inplace=True, drop=True)

# Look at correlations between non-object features that have >2 unique numbers.

drop_cols = [x for x,y in zip(df2.columns, list((df2.apply(lambda x: x.nunique()) <= 2) == True)) if y == True]

sns.pairplot(data=df2.drop(drop_cols, axis=1).select_dtypes(exclude=['object']))
# Define some data labels for easy acces.

x_lbl, y_lbl, z_lbl = "satisfaction_level", "last_evaluation", "average_montly_hours"

# Plot satisfaction vs. evaluation

fig = plt.figure()

plt.plot(df2[x_lbl], df2[y_lbl], 'ko')

# Visually, we see three very unusual clusters that are hard bounded

# in addition to some low-level noise.

# Just by visual inspection, we can come up with some rough cluster centers

cl_centers = np.array([[0.1, 0.88], # upper left

                       [0.41, 0.51], # lower middle, seems very dense

                       [0.83, 0.9]]) # upper right, sparse

[plt.plot(cl_centers[x][0], cl_centers[x][1], 'o', c='red') for x in range(len(cl_centers))]

plt.show()
# Perform KMeans clustering given qualitative mean center guesses

n_clusters = 3

km = KMeans(n_clusters=3).fit(df2[[x_lbl, y_lbl]])

# Cluster labels

labels = km.labels_

df2['labels'] = np.asarray(labels).astype(int)

# Plot KMeans clusters

fig = plt.figure(figsize=(12,8))

colors = ['red', 'green', 'blue']

x_lbl, y_lbl = "satisfaction_level", "last_evaluation"

[plt.scatter(df2[df2['labels'] == x][x_lbl],

             df2[df2['labels'] == x][y_lbl],

             c=colors[x], label="Cluster %s" % x) for x in range(n_clusters)]

plt.xlabel(x_lbl); plt.ylabel(y_lbl)

plt.legend(loc=3, ncol=3, bbox_to_anchor=(0., 1.02, 1., .102), mode='expand')

plt.show()
# Scikit learn's bandwidth estimator doesn't play well with pandas dataframes,

# so we have to cast it as a numpy array.

X = np.array(df2[[x_lbl, y_lbl]]) 

# Quantile (median pairwise distance) set to be just a bit smaller than default (0.3)

bandwidth = estimate_bandwidth(X, quantile=0.2)

# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)

ms.fit(X)

labels = ms.labels_

cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)

n_clusters_ = len(labels_unique)

df2['labels2'] = labels



# Plot the mean-shifted clusters and the fractional size of those clusters

# relative to the total size of the 'left' dataset

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,10))

ax[0].scatter(df2[df2['labels2'] == 0][x_lbl], df2[df2['labels2'] == 0][y_lbl], c='red', label="Cluster 0")

ax[0].scatter(df2[df2['labels2'] == 1][x_lbl], df2[df2['labels2'] == 1][y_lbl], c='blue', label="Cluster 1")

ax[0].scatter(df2[df2['labels2'] == 2][x_lbl], df2[df2['labels2'] == 2][y_lbl], c='green', label="Cluster 2")

ax[0].scatter(df2[df2['labels2'].isin(np.arange(3,10))][x_lbl],

              df2[df2['labels2'].isin(np.arange(3,10))][y_lbl],

              alpha=0.2, c='grey', label='Clusters 3-9')

ax[0].set_xlabel(x_lbl)

ax[0].set_ylabel(y_lbl)

ax[0].legend(loc=3, ncol=4, bbox_to_anchor=(0., 1.02, 1., .102), mode='expand')

# Plot showing size of clusters relative to data subset of workers who left

ax[1].plot(sorted(df2['labels2'].unique()),

           df2.groupby('labels2')['labels2'].count()/df2.shape[0], 'ko-')

ax2 = ax[1].twinx()

ax2.plot(sorted(df2['labels2'].unique()), df2.groupby('labels2')['labels2'].count(), 'ko-')

ax[1].set_xlabel("Cluster number"); ax[1].set_ylabel("Fraction of total 'left' data")

ax2.set_ylabel("# points in cluster")

plt.tight_layout()

plt.show()
# Create 3D plot of:

# x=satisfaction_levels

# y=last_evaluation

# z=average_montly_hours

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

colors = ["red", "blue", "green"]

for i in range(3):

    ax.scatter(df2[df2['labels2'] == i][x_lbl],

               df2[df2['labels2'] == i][y_lbl],

               df2[df2['labels2'] == i][z_lbl],

               c=colors[i], label='Cluster %s' % i)

ax.set_xlabel(x_lbl)

ax.set_ylabel(y_lbl)

ax.set_zlabel(z_lbl)

plt.legend()

plt.tight_layout()

plt.show()
# Perform PCA on 3D dataset of only clusters 0-2, reduce to 2D

X = df2[df2['labels2']][[x_lbl, y_lbl, z_lbl]]

pca = decomposition.PCA(n_components=2)

pca.fit(X)

X = pca.transform(X)

df2['pca_1'], df2['pca_2'] = X[:,0], X[:,1]

plt.plot(df2[df2['labels2']==0]['pca_1'], df2[df2['labels2']==0]['pca_2'], 'ro')

plt.plot(df2[df2['labels2']==1]['pca_1'], df2[df2['labels2']==1]['pca_2'], 'bo')

plt.plot(df2[df2['labels2']==2]['pca_1'], df2[df2['labels2']==2]['pca_2'], 'go')

plt.xlabel("PC1"); plt.ylabel("PC2")

plt.show()
# Cluster 0

X0 = np.array(df2[df2['labels2'] == 0][[x_lbl, y_lbl, z_lbl]])

# Cluster 1

X1 = np.array(df2[df2['labels2'] == 1][[x_lbl, y_lbl, z_lbl]])

# Cluster 2

X2 = np.array(df2[df2['labels2'] == 2][[x_lbl, y_lbl, z_lbl]])





# PCA on cluster 0

pca = decomposition.PCA(n_components=2)

pca.fit(X0)

# Cluster 0 is much more "dense" than 1 and 2, we can simply spread out the

# datapoints by scaling PC1 and PC2 by it's number of points relative to the

# mean of the number of points in clusters 1 and 2.

X0 = pca.transform(X0)*(X0.shape[0]/((X1.shape[0] + X2.shape[0])/2.)) # scale cluster 0 data

# PCA on cluster 1

pca = decomposition.PCA(n_components=2)

pca.fit(X1)

X1 = pca.transform(X1)

# PCA on cluster 2

pca = decomposition.PCA(n_components=2)

pca.fit(X2)

X2 = pca.transform(X2)

# Plot normalized and scaled values

fig = plt.figure(figsize=(6,8))

ax = fig.add_subplot(211)

ax.plot(X0[:,0], X0[:,1], '.', c=colors[0], label='Cluster 0')

ax.plot(X1[:,0], X1[:,1], '.', c=colors[1], label='Cluster 1')

ax.plot(X2[:,0], X2[:,1], '.', c=colors[2], label='Cluster 2')

ax.set_xlim(-100, 100)

ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

plt.legend()

# Plot in 3D with Z-axis as cluster number

ax = fig.add_subplot(212, projection='3d')

ax.scatter(X0[:,0], X0[:,1], 0, '.', c=colors[0], label='Cluster 0')

ax.scatter(X1[:,0], X1[:,1], 1, '.', c=colors[1], label='Cluster 1')

ax.scatter(X2[:,0], X2[:,1], 2, '.', c=colors[2], label='Cluster 2')

ax.set_xlim(-100, 100)

ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("Cluster #")

plt.legend()

plt.tight_layout()

plt.show()