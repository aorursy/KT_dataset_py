## 1.0 Call libraries

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
## 2. Read data

wh_data1= pd.read_csv("../input/2017.csv", header = 0)

wh_data1.describe
print("Dimension of dataset: wh.shape")

wh_data1.dtypes
## 3.1 Explore and scale

wh_data = wh_data1.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns

wh_data.head(2)
### 3.2 Scale the dataset



ss = StandardScaler()

ss.fit_transform(wh_data)
## 4 Visualization and Clustering

#### 4.1 Heatmap citing correlation

wh_data = wh_data1[['Happiness.Score','Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.', 'Freedom', 

          'Generosity','Trust..Government.Corruption.','Dystopia.Residual']] #Subsetting the data

cor = wh_data.corr()

sns.heatmap(cor, square = True)

plt.show()
#### 4.2 k-means clustering



def doKmeans(X, nclust=2):

    model = KMeans(nclust)

    model.fit(X)

    clust_labels = model.predict(X)

    cent = model.cluster_centers_

    return (clust_labels, cent)



clust_labels, cent = doKmeans(wh_data, 2)

kmeans = pd.DataFrame(clust_labels)

wh_data.insert((wh_data.shape[1]),'kmeans',kmeans)

#### Plot the clusters obtained using k means

fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(wh_data['Economy..GDP.per.Capita.'],wh_data['Trust..Government.Corruption.'],

                     c=kmeans[0],s=50)

ax.set_title('K-Means Clustering')

ax.set_xlabel('GDP per Capita')

ax.set_ylabel('Corruption')

plt.colorbar(scatter)
#### 4.3 Agglomerative Clustering



def doAgglomerative(X, nclust=2):

    model = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')

    clust_labels1 = model.fit_predict(X)

    return (clust_labels1)



clust_labels1 = doAgglomerative(wh_data, 2)

agglomerative = pd.DataFrame(clust_labels1)

wh_data.insert((wh_data.shape[1]),'agglomerative',agglomerative)
#### Plot the clusters obtained using Agglomerative clustering or Hierarchical clustering

fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(wh_data['Economy..GDP.per.Capita.'],wh_data['Trust..Government.Corruption.'],

                     c=agglomerative[0],s=50)

ax.set_title('Agglomerative Clustering')

ax.set_xlabel('GDP per Capita')

ax.set_ylabel('Corruption')

plt.colorbar(scatter)
#### 4.4 Affinity Propagation



def doAffinity(X):

    model = AffinityPropagation(damping = 0.5, max_iter = 250, affinity = 'euclidean')

    model.fit(X)

    clust_labels2 = model.predict(X)

    cent2 = model.cluster_centers_

    return (clust_labels2, cent2)



clust_labels2, cent2 = doAffinity(wh_data)

affinity = pd.DataFrame(clust_labels2)

wh_data.insert((wh_data.shape[1]),'affinity',affinity)
fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(wh_data['Economy..GDP.per.Capita.'],wh_data['Trust..Government.Corruption.'],

                     c=affinity[0],s=50)

ax.set_title('Affinity Clustering')

ax.set_xlabel('GDP per Capita')

ax.set_ylabel('Corruption')

plt.colorbar(scatter)
#### 4.5 Guassian Mixture Modelling



def doGMM(X, nclust=2):

    model = GaussianMixture(n_components=nclust,init_params='kmeans')

    model.fit(X)

    clust_labels3 = model.predict(X)

    return (clust_labels3)



clust_labels3 = doGMM(wh_data,2)

gmm = pd.DataFrame(clust_labels3)

wh_data.insert((wh_data.shape[1]),'gmm',gmm)
#Plotting the cluster obtained using GMM

fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(wh_data['Economy..GDP.per.Capita.'],wh_data['Trust..Government.Corruption.'],

                     c=gmm[0],s=50)

ax.set_title('Affinity Clustering')

ax.set_xlabel('GDP per Capita')

ax.set_ylabel('Corruption')

plt.colorbar(scatter)
## 5 Plotting maps for Global Score: Countrywise

### 5.1 Kmeans Algorithm    

wh_data.insert(0,'Country',wh_data1.iloc[:,0])

wh_data.iloc[:,[0,9,10,11,12]]

data = [dict(type='choropleth',

             locations = wh_data['Country'],

             locationmode = 'country names',

             z = wh_data['kmeans'],

             text = wh_data['Country'],

             colorbar = {'title':'Cluster Group'})]

layout = dict(title='Clustering of Countries based on K-Means',

              geo=dict(showframe = False,

                       projection = {'type':'Mercator'}))

map1 = go.Figure(data = data, layout=layout)

iplot(map1)
### 5.2 Agglomerative Clustering

data = [dict(type='choropleth',

             locations = wh_data['Country'],

             locationmode = 'country names',

             z = wh_data['agglomerative'],

             text = wh_data['Country'],

             colorbar = {'title':'Cluster Group'})]

layout = dict(title='Grouping of Countries based on Agglomerative Clustering',

              geo=dict(showframe = False, 

                       projection = {'type':'Mercator'}))

map2 = dict(data=data, layout=layout)

iplot(map2)
### 5.3 Affinity Propagation



data = [dict(type='choropleth',

             locations = wh_data['Country'],

             locationmode = 'country names',

             z = wh_data['affinity'],

             text = wh_data['Country'],

             colorbar = {'title':'Cluster Group'})]

layout = dict(title='Grouping of Countries based on Affinity Clustering',

              geo=dict(showframe = False, projection = {'type':'Mercator'}))

map3 = dict(data=data, layout=layout)

iplot(map3)
### 5.4 GMM

data = [dict(type='choropleth',

             locations = wh_data['Country'],

             locationmode = 'country names',

             z = wh_data['gmm'],

             text = wh_data['Country'],

             colorbar = {'title':'Cluster Group'})]

layout = dict(title='Grouping of Countries based on GMM clustering',

              geo=dict(showframe = False, projection = {'type':'Mercator'}))

map4 = dict(data=data, layout=layout)

iplot(map4)