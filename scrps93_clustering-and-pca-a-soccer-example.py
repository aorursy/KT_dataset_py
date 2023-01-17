from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns



print(os.listdir('../input'))
# We bring the functions necessary for PCA analysis

from sklearn.decomposition import PCA
# Full dataset

data = pd.read_csv('../input/data.csv')

data.head()
# Parameter list

variables = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

#variables = ['Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']



df = data[variables]

df = df.dropna(how='all') # We must drop all NA values to apply PCA

df = df.fillna(df.mean()) # We fill these values with the mean values
# Data visualization

df.head()
# In order to improve convergence, we standardize our data



stand_df = StandardScaler().fit_transform(df.loc[:,variables].values)
# Applying PCA analysis

pca = PCA(n_components=16)

pcafit = pca.fit(stand_df)

pcafeatures = pca.transform(stand_df)



features = range(pca.n_components_)



num_comp = 5

var_percent = sum(pca.explained_variance_ratio_[0:num_comp])

print("Total explained variance by the first %i components is %.5f" % (num_comp, var_percent))
# Percentage of variance explained per component

plt.bar(features, pca.explained_variance_ratio_)

plt.xlabel('Component number')

plt.ylabel('Percentage of variance')

plt.show()
principalComponents = pca.fit_transform(stand_df)



principalDf = pd.DataFrame(data = principalComponents)

principalDf = principalDf[[0, 1, 2, 3, 4]]

principalDf.columns = [['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]

principalDf.head()
g = sns.PairGrid(principalDf.loc[:,['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])

g.map_diag(plt.hist, histtype="step", linewidth=2)

g.map_offdiag(plt.scatter)



plt.show()
from sklearn import preprocessing

from scipy.cluster.hierarchy import dendrogram, linkage
import sys

sys.setrecursionlimit(10000)



dist_sin = linkage(principalDf.loc[:,['PC1', 'PC2', 'PC3', 'PC4', 'PC5']],method='single')

plt.figure(figsize=(20,8))

dendrogram(dist_sin, leaf_rotation=90)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("Dendogram: single method",fontsize=20)

plt.show()
dist_sin = linkage(principalDf.loc[:,['PC1', 'PC2', 'PC3', 'PC4', 'PC5']],method='single')

plt.figure(figsize=(20,8))

dendrogram(dist_sin, leaf_rotation=90, p=100, truncate_mode='lastp')

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("Dendogram: truncated single method",fontsize=20)

plt.show()
dist_sin = linkage(principalDf.loc[:,['PC1', 'PC2', 'PC3', 'PC4', 'PC5']],method='complete')

plt.figure(figsize=(20,8))

dendrogram(dist_sin, leaf_rotation=90)

plt.xlabel('Index')

plt.ylabel('Distance')

plt.suptitle("Dendogram: complete method",fontsize=20)

plt.show()