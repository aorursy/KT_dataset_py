## 1.0 Call libraries



import time                   # To time processes

import warnings               # To suppress warnings

import plotly.graph_objs as go

import os                     # For os related operations

import sys                    # For data size

import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulatio 

import matplotlib.pyplot as plt                   # For graphics





from sklearn import cluster, mixture              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset                  

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline

# 2. Read data

X= pd.read_csv("../input/2017.csv", header=0)

Country_df=pd.read_csv("../input/2017.csv", header=0)

#X= pd.read_csv("iris.csv", header = 0)

Country_df=Country_df['Country' ]
# 3. Explore and scale

X.columns.values

X.shape                 # 155 X 12

X = X.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns

X.head(2)

X.dtypes

X.info


# 3.1 Normalize dataset for easier parameter selection

# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

#    Standardize features by removing the mean

#      and scaling to unit variance

# 3.1.2 Instantiate scaler object

ss = StandardScaler()

# 3.1.3 Use ot now to 'fit' &  'transform'

ss.fit_transform(X)


#### 5. Begin Clustering   

                                  

# 5.1 How many clusters

#     NOT all algorithms require this parameter

n_clusters = 2 
##  KMeans

# Ref: http://scikit-learn.org/stable/modules/clustering.html#k-means                                  

# KMeans algorithm clusters data by trying to separate samples in n groups

#  of equal variance, minimizing a criterion known as the within-cluster

#   sum-of-squares.  





##  Mean Shift

# http://scikit-learn.org/stable/modules/clustering.html#mean-shift

# This clustering aims to discover blobs in a smooth density of samples.

#   It is a centroid based algorithm, which works by updating candidates

#    for centroids to be the mean of the points within a given region.

#     These candidates are then filtered in a post-processing stage to

#      eliminate near-duplicates to form the final set of centroids.

# Parameter: bandwidth dictates size of the region to search through. 



##  Mini Batch K-Means

#  Ref: 

#     http://scikit-learn.org/stable/modules/clustering.html#mini-batch-k-means

#  Similar to kmeans but clustering is done in batches to reduce computation time





##  Spectral clustering

# http://scikit-learn.org/stable/modules/clustering.html#spectral-clustering    

# SpectralClustering does a low-dimension embedding of the affinity matrix

#  between samples, followed by a KMeans in the low dimensional space. It

#   is especially efficient if the affinity matrix is sparse.

#   SpectralClustering requires the number of clusters to be specified.

#     It works well for a small number of clusters but is not advised when 

#      using many clusters.



## DBSCAN

# http://scikit-learn.org/stable/modules/clustering.html#dbscan

#   The DBSCAN algorithm views clusters as areas of high density separated

#    by areas of low density. Due to this rather generic view, clusters found

#     by DBSCAN can be any shape, as opposed to k-means which assumes that

#      clusters are convex shaped.    

#    Parameter eps decides the incremental search area within which density

#     should be same



# Affinity Propagation

# Ref: http://scikit-learn.org/stable/modules/clustering.html#affinity-propagation    

# Creates clusters by sending messages between pairs of samples until convergence.

#  A dataset is then described using a small number of exemplars, which are

#   identified as those most representative of other samples. The messages sent

#    between pairs represent the suitability for one sample to be the exemplar

#     of the other, which is updated in response to the values from other pairs. 

#       Two important parameters are the preference, which controls how many

#       exemplars are used, and the damping factor which damps the responsibility

#        and availability messages to avoid numerical oscillations when updating

#         these messages.



##  Birch

# http://scikit-learn.org/stable/modules/clustering.html#birch    

# The Birch builds a tree called the Characteristic Feature Tree (CFT) for the

#   given data and clustering is performed as per the nodes of the tree



# Gaussian Mixture modeling

#  http://203.122.28.230/moodle/course/view.php?id=6&sectionid=11#section-3

#  It treats each dense region as if produced by a gaussian process and then

#  goes about to find the parameters of the process
# Define different Clustering Techniques



def Kmeans():

    # 5.1 Instantiate object

    km = cluster.KMeans(n_clusters =n_clusters )

    # 5.2.1 Fit the object to perform clustering

    km_result = km.fit_predict(X)

    return km_result



def MeanShift():

    bandwidth = 0.1 

    # 6.2 No of clusters are NOT predecided

    ms = cluster.MeanShift(bandwidth=bandwidth)

    # 6.3

    ms_result = ms.fit_predict(X)

    return ms_result



def MiniBatch():

    two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)

    two_means_result = two_means.fit_predict(X)

    return two_means_result



def Spectral():

    spectral = cluster.SpectralClustering(n_clusters=n_clusters)

    sp_result= spectral.fit_predict(X)

    return sp_result



def DBScan():

    eps = 0.3

    dbscan = cluster.DBSCAN(eps=eps)

    db_result= dbscan.fit_predict(X)

    return db_result



def Affinity():

    damping = 0.9

    preference = -200

    affinity_propagation = cluster.AffinityPropagation(

        damping=damping, preference=preference)

    affinity_propagation.fit(X)

    ap_result = affinity_propagation .predict(X)

    return ap_result



def Birch():

    birch = cluster.Birch(n_clusters=n_clusters)

    birch_result = birch.fit_predict(X)

    return birch_result



def Gaussian():

    gmm = mixture.GaussianMixture( n_components=n_clusters, covariance_type='full')

    gmm.fit(X)

    gmm_result = gmm.predict(X)

    return gmm_result

    

# Create dataframe for result

Result_DataFrame = pd.DataFrame(columns=['Country','KMeans','MeanShift','Minibatch','Spectral','DBSCAN','Affinity','Birch','Gaussian'])

Result_DataFrame.loc[:,"Country"]=Country_df

# Switcher similar to switch case statement

switcher = {

        0: Kmeans,

        1: MeanShift,

        2: MiniBatch,

        3: Spectral,

        4: DBScan,

        5: Affinity,

        6: Birch,

        7: Gaussian

    }

Cluster_Array=[0,1,2,3,4,5,6,7]

Cluster_list=['KMeans','MeanShift','Minibatch','Spectral','DBSCAN','Affinity','Birch','Gaussian']

def Cluster_tech(argument):

    # Get the function from switcher dictionary

    func = switcher.get(argument, "nothing")

    # Execute the function

    return func()



for i in Cluster_Array:

    Result_DataFrame.iloc[:,i+1]=Cluster_tech(i)

    plt.subplot(4, 2, i+1)

    plt.title(Cluster_list[i])

    plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=Cluster_tech(i))

    

plt.subplots_adjust(bottom=-0.5, top=2.0)

Result_DataFrame

km = cluster.KMeans(n_clusters =n_clusters )

km_result = km.fit_predict(X)

data = dict(type = 'choropleth', 

           locations = Country_df,

           locationmode = 'country names',

           z = km_result, 

           text = Country_df,

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'K-Means Clustering', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap1 = go.Figure(data = [data], layout=layout)

iplot(choromap1)