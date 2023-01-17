import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

from sklearn import datasets

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn import cluster, mixture # For clustering 

import types

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline

df = pd.read_csv("../input/2017.csv", header=0)

df.head(2)

df.dtypes
data = dict(type = 'choropleth', 

           locations = df['Country'],

           locationmode = 'country names',

           z = df['Happiness.Rank'], 

           text = df['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'Global Happiness', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
# as we know now features like health economy and dystpia residue affect the most 

# so let's do indivisual test with happiness and the 3 features 

year = df['Happiness.Rank']

plt.scatter(df['Economy..GDP.per.Capita.'],year)

plt.scatter(df['Health..Life.Expectancy.'],year)

plt.scatter(df['Dystopia.Residual'],year)

plt.figure(figsize=(12,10))

plt.show()
country=df[df.columns[0]]

data= df.iloc[:,2:]
def normalizedData(x):

    normalised = StandardScaler()

    normalised.fit_transform(x)

    return(x)
data = normalizedData(data)
n_clusters=2

def Kmeans(x, y):

    km= cluster.KMeans(x)

    km_result=km.fit_predict(y)

    return(km_result)
km_result = Kmeans(2,data)

data['Kmeans'] = pd.DataFrame(km_result)

plt.scatter(data.iloc[:, 0], data.iloc[:, 9],  c=km_result)

plt.show()  
dataset=pd.concat([data,country],axis=1)

dataset.head(3)
dataPlot = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['Kmeans'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Kmeans Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [dataPlot], layout=layout)

iplot(choromap3) 
def MeanShift(x,y):

    ms=cluster.MeanShift(x)

    ms_result=ms.fit_predict(y)

    return(ms_result)
ms_result=MeanShift(0.1,data)

data['MeanShift'] = pd.DataFrame(ms_result)

plt.scatter(data.iloc[:, 0], data.iloc[:, 9],  c=ms_result)

plt.show()
dataset=pd.concat([data,country],axis=1)

dataset.head(3)
dataPlot = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['MeanShift'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'MeanShift  Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [dataPlot], layout=layout)

iplot(choromap3) 
def MiniKmeans(x, y):

    mb= cluster.MiniBatchKMeans(x)

    mb_result=mb.fit_predict(y)

    return(mb_result)
mb_result = MiniKmeans(3,data)

data['MiniKmeans'] = pd.DataFrame(mb_result)

plt.scatter(data.iloc[:, 0], data.iloc[:, 9],  c=mb_result)

plt.show()  
dataset=pd.concat([data,country],axis=1)

dataPlot = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['MiniKmeans'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'MiniKmeans  Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [dataPlot], layout=layout)

iplot(choromap3) 
spectral = cluster.SpectralClustering(n_clusters=n_clusters)

sp_result= spectral.fit_predict(data)

data['spectral'] = pd.DataFrame(sp_result)

plt.scatter(data.iloc[:, 0], data.iloc[:, 9],  c=sp_result)

plt.show()
dataset=pd.concat([data,country],axis=1)

dataPlot = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['spectral'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'spectral  Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [dataPlot], layout=layout)

iplot(choromap3) 
def Dbscan(x, y):

    db=cluster.DBSCAN(eps=x)

    db_result=db.fit_predict(y)

    return(db_result)
db_result = Dbscan(0.3,data)

data['Dbscan'] = pd.DataFrame(db_result)

plt.scatter(data.iloc[:, 0], data.iloc[:, 9],  c=db_result)

plt.show() 
dataset=pd.concat([data,country],axis=1)

dataPlot = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['Dbscan'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Dbscan  Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [dataPlot], layout=layout)

iplot(choromap3) 
def Affinity(x, y,z):

    ap=cluster.AffinityPropagation(damping=x, preference=y)

    ap_result=ap.fit_predict(z)

    return(ap_result)

ap_result = Affinity(0.9,-200,data)

data['Affinity'] = pd.DataFrame(ap_result)

plt.scatter(data.iloc[:, 0], data.iloc[:, 9],  c=ap_result)

plt.show() 
dataset=pd.concat([data,country],axis=1)

dataPlot = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['Affinity'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Affinity  Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [dataPlot], layout=layout)

iplot(choromap3) 
def Bir(x, y):

    bi=cluster.Birch(n_clusters=x)

    bi_result=bi.fit_predict(y)

    return(bi_result)
bi_result = Bir(3,data)

data['Bir'] = pd.DataFrame(bi_result)

plt.scatter(data.iloc[:, 0], data.iloc[:, 9],  c=bi_result)

plt.show() 
dataset=pd.concat([data,country],axis=1)

dataPlot = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['Bir'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Birch  Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [dataPlot], layout=layout)

iplot(choromap3) 
def gmm(x, y):

    gm=mixture.GaussianMixture(n_components=x,covariance_type='full')

    gm.fit(y)

    gm_result=gm.predict(y)

    return(gm_result)
gm_result = gmm(3,data)

data['gmm'] = pd.DataFrame(gm_result)

plt.scatter(data.iloc[:, 0], data.iloc[:, 9],  c=gm_result)

plt.show()
dataset=pd.concat([data,country],axis=1)

dataPlot = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['gmm'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Gaussian  Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [dataPlot], layout=layout)

iplot(choromap3) 