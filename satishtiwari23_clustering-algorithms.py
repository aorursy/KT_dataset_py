## 1 Call libraries

import numpy as np                   # Data manipulation

import pandas as pd                  # DataFrame manipulation

import time                          # To time processes 

import warnings                      # To suppress warnings

import matplotlib.pyplot as plt      # For Graphics

import seaborn as sns

from sklearn import cluster, mixture # For clustering 

from sklearn.preprocessing import StandardScaler



import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline

warnings.filterwarnings('ignore')
# 2. Read data

WHR_Data= pd.read_csv("../input/2017.csv", header = 0)

WHR_Data.head(2)
# 2. Explore and scale data

X = WHR_Data.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns

X.dtypes
# 4 Normalize the data



# 4.1 Instantiate scaler object

ss = StandardScaler()

# 4.2 Use ot now to 'fit' &  'transform'

ss.fit_transform(X)
# Thisclass contains functions for all clustering methods

class ClustringAlgorithms(object) :

    n_clusters = 2      # No of clusters. To use with techniques which needs this as input

    bandwidth = 0.1     # To use with Mean Shift technique

    eps = 0.3           # To use with DbScan technique for incremental area density

    damping = 0.9       # To use with Affinity Propogation technique

    preference = -200   # To use with Affinity Propogation technique 

      

    # KMeans algorithm clusters data by trying to separate samples in n groups

    # of equal variance, minimizing a criterion known as the within-cluster sum-of-squares.

    # Parameter: Dataset, No of clusters.

    def kmeans(self, X, n_clusters = n_clusters):

        km = cluster.KMeans(n_clusters)

        return km.fit_predict(X)

    

    # This clustering aims to discover blobs in a smooth density of samples.

    # It is a centroid based algorithm, which works by updating candidates

    # for centroids to be the mean of the points within a given region.

    # These candidates are then filtered in a post-processing stage to

    # eliminate near-duplicates to form the final set of centroids.

    # Parameter: Dataset, bandwidth dictates size of the region to search through.

    def meanshift(self, X, bandwidth=bandwidth):

        ms = cluster.MeanShift(bandwidth)

        return  ms.fit_predict(X)

    

    # Similar to kmeans but clustering is done in batches to reduce computation time

    # Parameter: Dataset,No of clusters.

    def minibatchkmeans(self, X, n_clusters = n_clusters):

        two_means = cluster.MiniBatchKMeans(n_clusters)

        return two_means.fit_predict(X)

   

    def spectral(self, X, n_clusters = n_clusters):

        sp = cluster.SpectralClustering(n_clusters)

        return sp.fit_predict(X)



    def dbscan(self, X, eps=eps):

        db = cluster.DBSCAN(eps)

        return db.fit_predict(X)

    

    def affinitypropagation(self, X, preference=preference, damping=damping):

        affinity_propagation =  cluster.AffinityPropagation(damping, preference)

        affinity_propagation.fit(X)

        return affinity_propagation.predict(X)

       

    def birch(self, X, n_clusters = n_clusters):

        birch = cluster.Birch(n_clusters)

        return birch.fit_predict(X)

   

    def gaussian_mixture(self, X, n_clusters = n_clusters):

        gmm = mixture.GaussianMixture(n_clusters, covariance_type='full')

        gmm.fit(X)

        return  gmm.predict(X)
def clusteringAlgoProcessing(dataSet):

    fig,ax = plt.subplots(4, 2, figsize=(10,10)) 

    clusterAlgo = ClustringAlgorithms()

    i = 0

    j=0

    listofClusterMethod = ['KMeans',"MeanShift","MiniBatchKmeans","DBScan","Spectral","Birch","Gaussian_Mixture"]

    for cm in listofClusterMethod :

        methodName = str(cm).lower()

        method = getattr(clusterAlgo, methodName)

        result = method(dataSet)

        dataSet[cm] = pd.DataFrame(result)

        ax[i,j].scatter(dataSet.iloc[:, 4], dataSet.iloc[:, 5],  c=result)

        ax[i,j].set_title(cm)

        j=j+1

        if( j % 2 == 0) :

            j= 0

            i=i+1

    plt.subplots_adjust(bottom=-0.5, top=1.5)

    plt.show()
clusteringAlgoProcessing(X)
X.insert(0,'Country',WHR_Data.iloc[:,0])

X.iloc[:,[0,11,12,13,14,15,16,17]]
### 5.1 Kmeans Algorithm  

data = dict(type = 'choropleth', 

           locations = X['Country'],

           locationmode = 'country names',

           z = X['KMeans'], 

           text = X['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'K-Means Clustering Visualization', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
data = dict(type = 'choropleth', 

           locations = X['Country'],

           locationmode = 'country names',

           z = X['Gaussian_Mixture'], 

           text = X['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Gaussian Mixture Clustering Visualization', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)