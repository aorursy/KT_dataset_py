#Call libraries

import time                   # To time processes

import warnings               # To suppress warnings

import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulatio 

import matplotlib.pyplot as plt  # For graphics

import os                     # For os related operations

import sys                    # For data size



from sklearn import cluster, mixture              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset

#%matplotlib inline            # To display plots inline

warnings.filterwarnings('ignore','UsageError')
os.chdir("../input")

df= pd.read_csv("2017.csv")



# Taken a 10% sample for analysis

X = df.sample(frac=0.1)



# Explore and scale dataset

X.columns.values

X.shape                 # 155 X 12

X = X.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns

X.head(2)

X.dtypes



# Normalization of dataset for easier parameter selection

ss = StandardScaler() #Instantiate scaler object

ss.fit_transform(X)
n_clusters = 2   #for K-means clustering,, Mini Batch K-Means. No of clusters to use

bandwidth = 0.1  #for Mean-Shift Clustering. bandwidth dictates size of the region to search through

eps = 0.3 #for DBSCAN Clustering. eps decides the incremental search area within which density should be same

damping = 0.9; preference = -200  #for Affinity Propagation. preference - controls how many exemplars are used

# damping factor - damps the responsibility and availability messages to avoid numerical oscillations when updating these messages

km = cluster.KMeans(n_clusters =n_clusters )

km_result = km.fit_predict(X)

ms = cluster.MeanShift(bandwidth=bandwidth)

ms_result = ms.fit_predict(X)

two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)

two_means_result = two_means.fit_predict(X)

spectral = cluster.SpectralClustering(n_clusters=n_clusters)

sp_result= spectral.fit_predict(X)

dbscan = cluster.DBSCAN(eps=eps)

db_result= dbscan.fit_predict(X)

affinity_propagation = cluster.AffinityPropagation(damping=damping, preference=preference) 

affinity_propagation.fit(X)

birch = cluster.Birch(n_clusters=n_clusters)

birch_result = birch.fit_predict(X)

gmm = mixture.GaussianMixture( n_components=n_clusters, covariance_type='full')

gmm.fit(X)
clustering_algorithms = (

        ('KMeans', km),

        ('MeanShift', ms),

        ('MiniBatchKMeans', two_means),

        ('SpectralClustering', spectral),

        ('DBSCAN', dbscan),

        ('AffinityPropagation', affinity_propagation),

        ('Birch', birch),

        ('GaussianMixture', gmm)

    )

result=algorithm.predict(X)

plot_num = 1 #for iteration

for name,algorithm in clustering_algorithms:

    y_pred = result

    y_pred = result

    plt.subplot(4, 2, plot_num)

    plt.scatter(X.iloc[:, 4], X.iloc[:, 5],c=result)

    plt.title(name, size=12)

    plot_num += 1

plt.show()
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



# Read data

whdata=pd.read_csv("2017.csv")

whdata = whdata.iloc[:, 2: ] 



# Instantiate scaler object

ss = StandardScaler()

# Use ot now to 'fit' &  'transform'

ss.fit_transform(whdata)



n_clusters = 2

km = cluster.KMeans(n_clusters =n_clusters )

km_result = km.fit_predict(whdata)



#Make a copy of the data set

whdata_map = whdata

whdata_map.head(2)

whdata.insert(0,'Country',df.iloc[:,0])

out=km_result



plt.subplot(4, 2, 1)

plt.scatter(whdata.iloc[:, 4], whdata.iloc[:, 5],  c=km_result)



whdata_map['clusters'] = out

data = dict(type = 'choropleth', 

           locations = whdata_map['Country'],

           locationmode = 'country names',

           z =  whdata_map['clusters'],

           text = whdata_map['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'World Happiness Using K Means Clustering Method', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)