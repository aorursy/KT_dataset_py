from sklearn.cluster import DBSCAN 

from sklearn.preprocessing import StandardScaler

from yellowbrick.cluster import KElbowVisualizer

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch

import sklearn.metrics as sk

from sklearn.cluster import KMeans

from math import sqrt

from sklearn import preprocessing

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import scipy

import numpy

import json

import sys

import csv

import os

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
#correltaion matrix/heatmap

df.corr()

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(df.corr())

plt.title('heat map')
x=df.drop(['Outcome'],axis=1)
#standardising data

x_scale = StandardScaler().fit_transform(x)
#elbow method to find n clusters

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)

    kmeans.fit(x_scale)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#check silhouette score

# Instantiate a scikit-learn K-Means model

model = KMeans(random_state=0)



# Instantiate the KElbowVisualizer with the number of clusters and the metric 

visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette', timings=False)



# Fit the data and visualize

visualizer.fit(x_scale)    

visualizer.poof()   
#applying kmeans algorith

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

pred_y = kmeans.fit_predict(x_scale)

plt.scatter(x_scale[:,0],x_scale[:,1],c=pred_y,cmap='viridis')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

plt.show()

#calculating davies bouldin score

sklearn.metrics.davies_bouldin_score(x_scale,pred_y)
#comparisons

x_kmeans=x.copy()

x_kmeans['labels']=pred_y

x_kmeans.groupby('labels').mean()

df.groupby('Outcome').mean()
#hierarchical clustering-plotting dendrogram

dendrogram = sch.dendrogram(sch.linkage(x_scale, method='ward'))
#applying agglomerative clustering algorithm

model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

model.fit_predict(x_scale)

labels = model.labels_

#plotting clusters on scatter plot

plt.figure(figsize=(10, 7))

plt.scatter(x_scale[labels==0, 0], x_scale[labels==0, 1], s=50, marker='o', color='red')

plt.scatter(x_scale[labels==1, 0], x_scale[labels==1, 1], s=50, marker='o', color='blue')

sklearn.metrics.davies_bouldin_score(x_scale,labels)
x_hrcl=x.copy()

x_hrcl['labels']=labels

x_hrcl.groupby('labels').mean()

df.groupby('Outcome').mean()

#density based clustering

db = DBSCAN(eps=3, min_samples=10).fit(x_scale) 

core_samples_mask = np.zeros_like(db.labels_, dtype=bool) 

core_samples_mask[db.core_sample_indices_] = True

labels_db = db.labels_ 

  

# Number of clusters in labels, ignoring noise if present. 

n_clusters_ = len(set(labels_db)) - (1 if -1 in labels_db else 0) 

  

print(labels_db) 

  

# Plot result 

  

# Black removed and is used for noise instead. 

unique_labels = set(labels_db) 

colors = ['y', 'b', 'g', 'r'] 

print(colors) 

for k, col in zip(unique_labels, colors): 

    if k == -1: 

        # Black used for noise. 

        col = 'k'

  

    class_member_mask = (labels_db == k) 

  

    xy = x_scale[class_member_mask & core_samples_mask] 

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 

                                      markeredgecolor='k',  

                                      markersize=6) 

  

    xy = x_scale[class_member_mask & ~core_samples_mask] 

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 

                                      markeredgecolor='k', 

                                      markersize=6) 

  

plt.title('number of clusters: %d' %n_clusters_) 

plt.show() 



sklearn.metrics.davies_bouldin_score(x_scale,labels_db )
x_db=x.copy()

x_db['labels']=labels_db

x_db.groupby('labels').mean()

df.groupby('Outcome').mean()

db=pd.DataFrame()

db['model']=['K-Means','Agglomerative Clustering','DBSCAN']

db['davies bouldin index']=[1.22,1.22,1.78]

db