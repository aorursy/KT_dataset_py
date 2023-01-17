# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt

%matplotlib inline

import matplotlib.cm as cm

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.preprocessing import StandardScaler  # For scaling dataset

from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering

from sklearn.mixture import GaussianMixture #For GMM clustering

from mpl_toolkits.mplot3d import Axes3D

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")


train_df=train_data.copy()

train_df=train_df.drop(["ID_code"], axis=1)

Y=train_df['target']

X = train_df.drop(["target"],axis=1)
#http://www.awesomestats.in/python-cluster-validation/



cluster_range = range( 1, 5 )

cluster_errors = []



for num_clusters in cluster_range:

    clusters = KMeans( num_clusters )

    clusters.fit( X )

    cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

plt.figure(figsize=(15,15))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
# Initializing KMeans

kmeans = KMeans(n_clusters=3)

# Fitting with inputs

kmeans = kmeans.fit(X)

# Predicting the clusters

labels = kmeans.predict(X)

# Getting the cluster centers

C = kmeans.cluster_centers_
fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(X.iloc[:, 0], X.iloc[:, 1],  c=Y)

ax.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=1000)
labels