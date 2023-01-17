import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import cluster, mixture # For clustering 

import types

from sklearn import metrics

import os

from sklearn.manifold import TSNE

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import seaborn as sns

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_blobs

from sklearn.cluster import MeanShift, estimate_bandwidth

init_notebook_mode(connected=True)

from pandas import read_csv

%matplotlib inline
def loadData():

    dataset = pd.read_csv("../input/2017.csv")

    dataset.head(5)

    return dataset



def normalizeData(dataset):

    # Instantiate Scaler Object 

    ss = StandardScaler()

    # Fit and transform 

    ss.fit_transform(dataset)

    dataset.head(10)

    country.head(10)

    return dataset

def univariateAnaly(dataset):

    sns.distplot(dataset['Happiness.Score'])

def heatmapAnaly(dataset):

    plt.figure(figsize=(10,8))

    datasetCorr=dataset.drop(['Happiness.Rank'],axis = 1)

    datasetCorr = datasetCorr.corr()

    sns.heatmap(datasetCorr, vmax=.8, square=True,annot=True,linewidths = .5, fmt='.2f',annot_kws={'size': 10} )

def missingValueChk(dataset):

    print(dataset.isnull().sum())

def detectOutlier(dataset):

    for i in range(3,dataset.shape[1]):

        quartile_1, quartile_3 = np.percentile(dataset[dataset.columns.values[i]], [25, 75])

        iqr = quartile_3 - quartile_1

        lower_bound = quartile_1 - (iqr * 1.5)

        upper_bound = quartile_3 + (iqr * 1.5)

        print(("Outlier in ",dataset.columns[i],":",np.where((dataset[dataset.columns.values[i]] > upper_bound) | (dataset[dataset.columns.values[i]] < lower_bound))))

def visualizeOutlier(dataset):

    sns.lmplot('Happiness.Score','Family',data=dataset,fit_reg=False) 

    sns.lmplot('Happiness.Score','Generosity',data=dataset,fit_reg=False)  

    sns.lmplot('Happiness.Score','Trust..Government.Corruption.',data=dataset,fit_reg=False)

    sns.lmplot('Happiness.Score','Dystopia.Residual',data=dataset,fit_reg=False)

def removeOutliers(col):

    quartile_1, quartile_3 = np.percentile(dataset[col], [25, 75])

    iqr = quartile_3 - quartile_1

    mean=dataset[col].mean()

    lower_bound = quartile_1 - (iqr * 1.5)

    upper_bound = quartile_3 + (iqr * 1.5)

    repeat=False

    for i in range(dataset.shape[0]):

        if((dataset[col].values[i]>upper_bound) | (dataset[col].values[i]<lower_bound)):

                dataset[col].values[i]=mean 

              #  print("repeat", repeat)

                repeat=True

    return repeat

def iterateOutliers(dataset):

    repeat=True

    while (repeat):

        detectOutlier(dataset)

        repeat1=removeOutliers('Family')

        repeat2=removeOutliers('Generosity')

        repeat3=removeOutliers('Trust..Government.Corruption.')

        repeat4=removeOutliers('Dystopia.Residual')

        repeat=repeat1 or repeat2 or repeat3 or repeat4

        detectOutlier(dataset)

    return dataset

def clustering(type,dataset,params):

    ds=dataset.drop(['Happiness.Rank','Happiness.Score'],axis=1)

    if(type=='kmeans'):

        kmeans = cluster.KMeans(n_clusters =params['nclusters'] )

        return kmeans.fit_predict(ds)

    if(type=='meanshift'):

        meanShift = cluster.MeanShift(bandwidth=params['bandwidth'])

        return  meanShift.fit_predict(ds)

    if(type=='minibatchkmeans'):

        miniBatch = cluster.MiniBatchKMeans(n_clusters=params['nclusters'])

        return miniBatch.fit_predict(ds)

    if(type=='dbscan'):

        dbscan = cluster.DBSCAN(eps=params['eps'])

        return dbscan.fit_predict(ds)

    if(type=='spectral'):

        spectral = cluster.SpectralClustering(n_clusters=params['nclusters'])

        return spectral.fit_predict(ds)

    if(type=='affinitypropagation'):

        affinityPropagation =  cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])

        affinityPropagation.fit(ds)

        return affinityPropagation.predict(ds)

    if(type=='birch'):

        birch = cluster.Birch(n_clusters=params['nclusters'])

        return birch.fit_predict(ds)

    if(type=='gaussian'):

        gmm = mixture.GaussianMixture( n_components=params['nclusters'], covariance_type='full')

        gmm.fit(ds)

        return  gmm.predict(ds)



def performClustering(clusterTypes,ds,params):

    fig,ax = plt.subplots(nrows,ncols, figsize=(10,10)) 

    i = 0

    j=0

    silhouetteDf=[]

    clusterName=[]

    for ct in clusterTypes['clustType'] :

        clusteringResult = clustering(ct,ds,params)

        clusterName.append(ct)

        ds[ct] = pd.DataFrame(clusteringResult)

        ax[i,j].scatter(ds.iloc[:, 4], ds.iloc[:, 5],  c=clusteringResult)

        ax[i,j].set_title(ct+"Clustering Result")

        j=j+1

        if( j % ncols == 0) :

            j= 0

            i=i+1

    plt.subplots_adjust(bottom=-0.5, top=1.5)

    plt.show()

    return ds



dataset=loadData()

country=dataset[dataset.columns[0]]

dataset=dataset.drop(['Country'],axis=1)

dataset=normalizeData(dataset)

dataset.head(5)
univariateAnaly(dataset)
heatmapAnaly(dataset)
missingValueChk(dataset)
detectOutlier(dataset)
visualizeOutlier(dataset)
dataset=iterateOutliers(dataset)
clusterTypes = {'clustType':['kmeans',"meanshift","minibatchkmeans","dbscan","spectral","affinitypropagation","birch","gaussian"]}

ncols = 2

nrows = 4

nclusters= 3

bandwidth = 0.1 

eps = 0.3

damping = 0.9

preference = -200

params = {'nclusters' :  nclusters, 'eps' : eps,'bandwidth' : bandwidth, 'damping' : damping, 'preference' : preference}

dataset.head(10)

dataset=performClustering(clusterTypes,dataset,params)
dataset=pd.concat([dataset,country],axis=1)

dataset.head(10)


data = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['Happiness.Score'], 

           text = dataset['Country'],

           colorbar = {'title':'Happiness Score'})

layout = dict(title = 'Global Happiness Score', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3) 
dataset.columns

data = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['kmeans'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Kmeans Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3) 
dataset.columns

data = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['meanshift'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'MeanShift Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3) 
dataset.columns

data = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['minibatchkmeans'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Minibatch Kmeans Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3) 