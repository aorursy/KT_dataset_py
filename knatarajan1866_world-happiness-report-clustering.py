# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# -*- coding: utf-8 -*-

## 1.0 Call libraries

import time                   # To time processes

import warnings               # To suppress warnings



import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulatio 

import matplotlib.pyplot as plt                   # For graphics



from sklearn import cluster, mixture, metrics, datasets              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset

from itertools import cycle, islice

import os                     # For os related operations

import sys                    # For data size

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

%matplotlib inline

# Define function for Reading Data 

def Happyreport():

    X = pd.read_csv("../input/2017.csv", header = 0)

    return X



def data_preprocessing(data):

    df = StandardScaler().fit_transform(data)

    df =pd.DataFrame(df, columns = X.columns)

    return df



def kmeans_clustering(nclusters, dataset ):

    km = cluster.KMeans(init = 'k-means++', n_clusters =n_clusters )

    km_result = km.fit_predict(df)

    return km_result, km

    

def mbkmeans_clustering(n_clusters, dataset):

    mbk = cluster.MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=100,

                      n_init=10, max_no_improvement=10, verbose=0)

    mbk_result = mbk.fit_predict(dataset)

    return mbk_result, mbk



def AffinityPropagation_clustering(damping, preference, data):

    af = cluster.AffinityPropagation(preference=preference, damping=damping).fit(df)

    ap_result = af.predict(df)

    return ap_result, af



def MeanShift_clustering(quantile, n_samples, dataset):

    bandwidth = cluster.estimate_bandwidth(df, quantile, n_samples)

    ms = cluster.MeanShift(bandwidth=bandwidth)

    ms_result = ms.fit_predict(df)

    return ms_result, ms



def Spectral_clustering(n_clusters):

    spectral = cluster.SpectralClustering(n_clusters=n_clusters)

    sp_result= spectral.fit_predict(df)

    return sp_result, spectral



def Agglomerative_Clustering(n_clusters, dataset, linkage):

    for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):

        if metric == 'cosine':

            cos_model = cluster.AgglomerativeClustering(n_clusters=n_clusters,

                                    linkage="average", affinity=metric)

            cos_result = cos_model.fit_predict(dataset)

        if metric == 'euclidean':

            ec_model = cluster.AgglomerativeClustering(n_clusters=n_clusters,

                                    linkage="average", affinity=metric)

            ec_result = ec_model.fit_predict(dataset)

        if metric == 'cityblock' :

            cb_model = cluster.AgglomerativeClustering(n_clusters=n_clusters,

                                                       linkage="average", affinity=metric)

            cb_result = cb_model.fit_predict(dataset)

    return cos_model, cos_result, ec_model, ec_result, cb_model, cb_result



def DBSCAN(dataset, eps):

    dbscan = cluster.DBSCAN(eps=eps)

    db_result= dbscan.fit_predict(X)

    return dbscan, db_result



def Birch(dataset, n_clusters):

    birch = cluster.Birch(n_clusters=n_clusters)

    birch_result = birch.fit_predict(X)

    return birch, birch_result



def GMM(dataset, n_clusters, covariance_type ):

    gmm = mixture.GaussianMixture( n_components=n_clusters, covariance_type='full')

    gmm.fit(X)

    gmm_result = gmm.predict(X)

    return gmm_result, gmm



#  Read Data 

X = Happyreport()

Y = Happyreport()

## Remove first 2 columns from the dataset

X = X.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns



## Data Preprocessing

df = data_preprocessing(X)

df_cpy = df.copy()

df = pd.DataFrame(df)



### Define Parameters

n_clusters = 2       

preference=-200

damping = 0.9

quantile=0.2

n_samples=155

linkage="average"

eps = 0.3

covariance_type='full'



#### KMeans Clustering

km_result, km = kmeans_clustering(n_clusters, df) 

df_cpy['Kmeans'] = pd.DataFrame(km_result)





#### Mini Batch K-Means

mbk_result, mbk = mbkmeans_clustering(n_clusters, df)

df_cpy['MBKmeans'] = pd.DataFrame(mbk_result )



# Compute Affinity Propagation

ap_result, af = AffinityPropagation_clustering(damping, preference, df)

df_cpy['Affinity Propagation'] = pd.DataFrame(ap_result )



### Meanshift

ms_result, ms = MeanShift_clustering(quantile, n_samples, df)

df_cpy['Meanshift'] = pd.DataFrame(ms_result)



## Spectral clustering

sp_result ,spectral = Spectral_clustering(n_clusters) 

df_cpy['Spectral'] = pd.DataFrame(sp_result)



## Agglomerative Clustering

cos_model, cos_result, ec_model, ec_result, cb_model, cb_result = Agglomerative_Clustering(n_clusters, df,linkage)

df_cpy['Agglomerative Cosine'] = pd.DataFrame(cos_result)

df_cpy['Agglomerative  Euclidean'] = pd.DataFrame(ec_result)

df_cpy['Agglomerative Cityblock'] = pd.DataFrame(cb_result)



## DBSCAN

dbscan, db_result = DBSCAN(df, eps)

df_cpy['DBScan'] = pd.DataFrame(db_result)



## Birch

birch, birch_result = Birch(df, n_clusters)

df_cpy['Birch'] = pd.DataFrame(birch_result)



## Gaussian Mixture modeling

gmm_result, gmm = GMM(df, n_clusters, covariance_type )

df_cpy['GMM'] = pd.DataFrame(gmm_result)





clustering_algorithms = (

         ('KMeans', km_result),

         ('MiniBatchKMeans', mbk_result),

         ('AffinityPropagation', ap_result),

         ('MeanShift', ms_result),

         ('SpectralClustering', sp_result),

         ('Agglo Cosine', cos_result),

         ('Agglo Euclidean', ec_result),

         ('Agglo Cityblock', cb_result),##

        ('DBSCAN', db_result),

        ('Birch', birch_result),

        ('GMM', gmm_result)

   )





fig,ax = plt.subplots(4,3, figsize=(5,5)) 

i = 0

j=0

for name, algorithm in clustering_algorithms:

    ax[i,j].scatter(df.iloc[:, 4], df.iloc[:, 5],  c=algorithm)

    ax[i,j].set_title(name)

    j=j+1

    if( j % 3 == 0) :

       j= 0

       i=i+1

plt.subplots_adjust(bottom=-0.5, top=1.5)

plt.show()



data = dict(type = 'choropleth', 

           locations = Y['Country'],

           locationmode = 'country names',

           z = df_cpy['Kmeans'], 

           text = Y['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'K-Means Clustering Visualization', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)







    
