## The python code below uses several clustering techniques on World Happiness Data (2017) from Kaggle. 

## This is a learning exercise to showcase the results obtained by various clustering algorithms via scatter plot and world map using plotly 

##    as here: https://plot.ly/python/choropleth-maps/
# Call libraries

import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulatio 

import matplotlib.pyplot as plt                   # For graphics



from sklearn import cluster, mixture              # For clustering

from sklearn.cluster import KMeans   # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset



import os                     # For os related operations

import sys                    # For data size

import time                   # To time processes

import warnings               # To suppress warnings



import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) #additional initialization step to plot offline in Jupyter Notebooks
# Read data

wh_data=pd.read_csv('../input/2017.csv')

wh_data.info()

#Make a copy of the data set

wh_data_map = wh_data
# Explore and scale

print(wh_data.columns.values)

print(wh_data.shape)                 

wh_data = wh_data.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns

print(wh_data.head(10))

print(wh_data.dtypes)

print(wh_data.info)
# Normalize dataset for easier parameter selection

#    Standardize features by removing the mean

#      and scaling to unit variance

# Instantiate scaler object

ss = StandardScaler()

# Use ot now to 'fit' &  'transform'

ss.fit_transform(wh_data)

#### Begin Clustering   

                                  

# How many clusters

n_clusters = 3        

                                  

## 1 KMeans

# KMeans algorithm clusters data by trying to separate samples in n groups

#  of equal variance, minimizing a criterion known as the within-cluster

#   sum-of-squares.                         



# Instantiate object

km = cluster.KMeans(n_clusters =n_clusters )



# Fit the object to perform clustering

km_result = km.fit_predict(wh_data)

out = km_result

# Draw scatter plot of two features colored by clusters

plt.subplot(4, 2, 1)

plt.scatter(wh_data.iloc[:, 4], wh_data.iloc[:, 5],  c=km_result)
wh_data_map['clusters'] = out

data = dict(type = 'choropleth', 

           locations = wh_data_map['Country'],

           locationmode = 'country names',

           z =  wh_data_map['clusters'],

           text = wh_data_map['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'World Happiness Using K Means Clustering Method', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
## 2 MeanShift

bandwidth = 0.1  



# No of clusters are NOT predecided

ms = cluster.MeanShift(bandwidth=bandwidth)



ms_result = ms.fit_predict(wh_data)

out = ms_result

plt.subplot(4, 2, 2)

plt.scatter(wh_data.iloc[:, 4], wh_data.iloc[:, 5],  c=ms_result)
# choropleth graph

wh_data_map['clusters'] = out

data = dict(type = 'choropleth', 

           locations = wh_data_map['Country'],

           locationmode = 'country names',

           z =  wh_data_map['clusters'],

           text = wh_data_map['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'World Happiness Using MeanShift Clustering Method', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
## 3 MiniBatchKMeans

# Scatter plot

wh_data_mkm = cluster.MiniBatchKMeans(n_clusters=3)

wh_data_mkm_result = wh_data_mkm.fit_predict(wh_data)

out = wh_data_mkm_result

plt.subplot(4, 2, 2)

plt.scatter(wh_data.iloc[:, 4], wh_data.iloc[:, 5],  c=wh_data_mkm_result)
# choropleth graph

wh_data_map['clusters'] = out

data = dict(type = 'choropleth', 

           locations = wh_data_map['Country'],

           locationmode = 'country names',

           z =  wh_data_map['clusters'],

           text = wh_data_map['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'World Happiness Using MiniBatchKMeans Clustering Method', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
## 4 Spectral Clustering

n_clusters = 3

spectral = cluster.SpectralClustering(n_clusters=n_clusters)

sp_result= spectral.fit_predict(wh_data)

out = sp_result



plt.subplot(4, 2, 4)

plt.scatter(wh_data.iloc[:, 4], wh_data.iloc[:, 5],  c=sp_result)
# choropleth graph

wh_data_map['clusters'] = out

data = dict(type = 'choropleth', 

           locations = wh_data_map['Country'],

           locationmode = 'country names',

           z =  wh_data_map['clusters'],

           text = wh_data_map['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'World Happiness Using Spectral Clustering Method', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
## 5 Affinity Propagation

# Scatter plot

damping = 0.9

preference = -200



# No of clusters are NOT predecided

affinity_propagation = cluster.AffinityPropagation(damping=damping, preference=preference)

affinity_propagation.fit(wh_data)

ap_result = affinity_propagation .predict(wh_data)

out = ap_result

plt.subplot(4, 2, 6)

plt.scatter(wh_data.iloc[:, 4], wh_data.iloc[:, 5],  c=ap_result)

# choropleth graph

wh_data_map['clusters'] = out

data = dict(type = 'choropleth', 

           locations = wh_data_map['Country'],

           locationmode = 'country names',

           z =  wh_data_map['clusters'],

           text = wh_data_map['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'World Happiness Using Affinity Propagation Clustering Method', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
## 6 DBSCAN

eps = 0.3



# No of clusters are NOT predecided

dbscan = cluster.DBSCAN(eps=eps)

db_result= dbscan.fit_predict(wh_data)

out = db_result



plt.subplot(4, 2, 5)

plt.scatter(wh_data.iloc[:, 4], wh_data.iloc[:, 5], c=db_result)
# choropleth graph

wh_data_map['clusters'] = out

data = dict(type = 'choropleth', 

           locations = wh_data_map['Country'],

           locationmode = 'country names',

           z =  wh_data_map['clusters'],

           text = wh_data_map['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'World Happiness Using DBScan Clustering Method', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
## 7 Birch

n_clusters = 3

birch = cluster.Birch(n_clusters=n_clusters)

birch_result = birch.fit_predict(wh_data)

out = birch_result



plt.subplot(4, 2, 7)

plt.scatter(wh_data.iloc[:, 4], wh_data.iloc[:, 5],  c=birch_result)
# choropleth graph

wh_data_map['clusters'] = out

data = dict(type = 'choropleth', 

           locations = wh_data_map['Country'],

           locationmode = 'country names',

           z =  wh_data_map['clusters'],

           text = wh_data_map['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'World Happiness Using Birch Clustering Method', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
## 8 Gaussian Mixture Modeling (GMM)

n_clusters = 3

gmm = mixture.GaussianMixture( n_components=n_clusters, covariance_type='full')

gmm.fit(wh_data)



gmm_result = gmm.predict(wh_data)

out = gmm_result



plt.subplot(4, 2, 8)

plt.scatter(wh_data.iloc[:, 4], wh_data.iloc[:, 5],  c=gmm_result)
# choropleth graph

wh_data_map['clusters'] = out

data = dict(type = 'choropleth', 

           locations = wh_data_map['Country'],

           locationmode = 'country names',

           z =  wh_data_map['clusters'],

           text = wh_data_map['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'World Happiness Using Gaussian Mixture Clustering Method', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)