#Call required libraries

import time                   # To time processes

import warnings               # To suppress warnings



import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulatio 

import matplotlib.pyplot as plt                   # For graphics

import seaborn as sns

import plotly.plotly as py #For World Map

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



from sklearn.preprocessing import StandardScaler  # For scaling dataset

from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering

from sklearn.mixture import GaussianMixture #For GMM clustering



import os                     # For os related operations

import sys                    # For data size

wh = pd.read_csv("../input/2017.csv") #Read the dataset

wh.describe()
print("Dimension of dataset: wh.shape")

wh.dtypes
wh1 = wh[['Happiness.Score','Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.', 'Freedom', 

          'Generosity','Trust..Government.Corruption.','Dystopia.Residual']] #Subsetting the data

cor = wh1.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
#Ref: https://plot.ly/python/choropleth-maps/

data = dict(type = 'choropleth', 

           locations = wh['Country'],

           locationmode = 'country names',

           z = wh['Happiness.Score'], 

           text = wh['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'Happiness Index 2017', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
#Scaling of data

ss = StandardScaler()

ss.fit_transform(wh1)
#K means Clustering 

def doKmeans(X, nclust=2):

    model = KMeans(nclust)

    model.fit(X)

    clust_labels = model.predict(X)

    cent = model.cluster_centers_

    return (clust_labels, cent)



clust_labels, cent = doKmeans(wh1, 2)

kmeans = pd.DataFrame(clust_labels)

wh1.insert((wh1.shape[1]),'kmeans',kmeans)





#Plot the clusters obtained using k means

fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],

                     c=kmeans[0],s=50)

ax.set_title('K-Means Clustering')

ax.set_xlabel('GDP per Capita')

ax.set_ylabel('Corruption')

plt.colorbar(scatter)
def doAgglomerative(X, nclust=2):

    model = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')

    clust_labels1 = model.fit_predict(X)

    return (clust_labels1)



clust_labels1 = doAgglomerative(wh1, 2)

agglomerative = pd.DataFrame(clust_labels1)

wh1.insert((wh1.shape[1]),'agglomerative',agglomerative)
#Plot the clusters obtained using Agglomerative clustering or Hierarchical clustering

fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],

                     c=agglomerative[0],s=50)

ax.set_title('Agglomerative Clustering')

ax.set_xlabel('GDP per Capita')

ax.set_ylabel('Corruption')

plt.colorbar(scatter)
def doAffinity(X):

    model = AffinityPropagation(damping = 0.5, max_iter = 250, affinity = 'euclidean')

    model.fit(X)

    clust_labels2 = model.predict(X)

    cent2 = model.cluster_centers_

    return (clust_labels2, cent2)



clust_labels2, cent2 = doAffinity(wh1)

affinity = pd.DataFrame(clust_labels2)

wh1.insert((wh1.shape[1]),'affinity',affinity)
#Plotting the cluster obtained using Affinity algorithm

fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],

                     c=affinity[0],s=50)

ax.set_title('Affinity Clustering')

ax.set_xlabel('GDP per Capita')

ax.set_ylabel('Corruption')

plt.colorbar(scatter)
def doGMM(X, nclust=2):

    model = GaussianMixture(n_components=nclust,init_params='kmeans')

    model.fit(X)

    clust_labels3 = model.predict(X)

    return (clust_labels3)



clust_labels3 = doGMM(wh1,2)

gmm = pd.DataFrame(clust_labels3)

wh1.insert((wh1.shape[1]),'gmm',gmm)
#Plotting the cluster obtained using GMM

fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],

                     c=gmm[0],s=50)

ax.set_title('Affinity Clustering')

ax.set_xlabel('GDP per Capita')

ax.set_ylabel('Corruption')

plt.colorbar(scatter)
wh1.insert(0,'Country',wh.iloc[:,0])

wh1.iloc[:,[0,9,10,11,12]]

data = [dict(type='choropleth',

             locations = wh1['Country'],

             locationmode = 'country names',

             z = wh1['kmeans'],

             text = wh1['Country'],

             colorbar = {'title':'Cluster Group'})]

layout = dict(title='Clustering of Countries based on K-Means',

              geo=dict(showframe = False,

                       projection = {'type':'Mercator'}))

map1 = go.Figure(data = data, layout=layout)

iplot(map1)
data = [dict(type='choropleth',

             locations = wh1['Country'],

             locationmode = 'country names',

             z = wh1['agglomerative'],

             text = wh1['Country'],

             colorbar = {'title':'Cluster Group'})]

layout = dict(title='Grouping of Countries based on Agglomerative Clustering',

              geo=dict(showframe = False, 

                       projection = {'type':'Mercator'}))

map2 = dict(data=data, layout=layout)

iplot(map2)
data = [dict(type='choropleth',

             locations = wh1['Country'],

             locationmode = 'country names',

             z = wh1['affinity'],

             text = wh1['Country'],

             colorbar = {'title':'Cluster Group'})]

layout = dict(title='Grouping of Countries based on Affinity Clustering',

              geo=dict(showframe = False, projection = {'type':'Mercator'}))

map3 = dict(data=data, layout=layout)

iplot(map3)
data = [dict(type='choropleth',

             locations = wh1['Country'],

             locationmode = 'country names',

             z = wh1['gmm'],

             text = wh1['Country'],

             colorbar = {'title':'Cluster Group'})]

layout = dict(title='Grouping of Countries based on GMM clustering',

              geo=dict(showframe = False, projection = {'type':'Mercator'}))

map4 = dict(data=data, layout=layout)

iplot(map4)