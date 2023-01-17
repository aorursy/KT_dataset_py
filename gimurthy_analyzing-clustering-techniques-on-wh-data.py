import numpy as np                                # Data manipulation

import pandas as pd                               # Dataframe manipulation 

import matplotlib.pyplot as plt                   # For graphics

import seaborn as sns                             # For graphics

from sklearn import cluster, mixture, metrics     # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset



# imports required for charting map

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



import os          # For os related operations

#import sys         # For data size

#import time        # To time processes

import warnings    # To suppress warnings



# Ignore warnings

warnings.filterwarnings("ignore")



# To show the graphs inline

%matplotlib inline
whr_data = pd.read_csv("../input/2017.csv", header=0)
whr_data.head()
# 3.1 Prepare Dataset for clustering analysis.

#   Removing the columns Country, Happiness.Rank from the analysis

#   Removing Happiness.Score, Whisker.high, Whisker.low as well since they are 

#   deduced by the rest of the columns. 

whr_data_for_clus = whr_data.iloc[ :, 5:]

whr_data_for_clus.head(3)
# 3.2 Center and scale the dataset

#   Using StandardScaler function, which Standardize features by removing the 

#   mean and scaling to unit variance..

# 3.2.2 Instantiate scaler object

ss = StandardScaler()

# 3.2.3 Use ot now to 'fit' &  'transform'

ss.fit_transform(whr_data_for_clus)
# 4.1.1 Define the constant parameters

n_clusters = 2      # No of clusters. To use with techniques which needs this as input

bandwidth = 0.1     # To use with Mean Shift technique

eps = 0.3           # To use with DbScan technique for incremental area density

damping = 0.9       # To use with Affinity Propogation technique

preference = -200   # To use with Affinity Propogation technique 

metric='euclidean'  # To use for silhouetter coefficient calculation to determing the right number of clusters



# 4.1.2 Define the Clustering technique Data Frame to use in looping thru the various techniques

cluster_dist = {'Technique' : ['K-means', 'Mean Shift', 'Mini Batch K-Means', 'Spectral', 'DBSCAN', 'Affinity Propagation', 'Birch', 'Gaussian Mixture Modeling'],

                'FunctionName' : ['Kmeans_Technique', 'MeanShift_Technique', 'MiniKmean_Technique', 'Spectral_Technique', 'Dbscan_Technique', 'AffProp_Technique', 'Birch_Technique', 'Gmm_Technique']}

cluster_df = pd.DataFrame(cluster_dist)





# 4.2 Define the functions 

#   4.2.1 Define the function for clustering thru K-means

def Kmeans_Technique(ds):

    """

    Ref: http://scikit-learn.org/stable/modules/clustering.html#k-means   

    """

    km = cluster.KMeans(n_clusters=n_clusters)

    return km.fit_predict(ds)



#   4.2.2 Define the function for clustering thru Mean Shift

def MeanShift_Technique(ds):

    """

    Ref: http://scikit-learn.org/stable/modules/clustering.html#mean-shift

    """

    ms = cluster.MeanShift(bandwidth=bandwidth)

    return ms.fit_predict(ds)



#   4.2.3 Define the function for clustering thru Mini Batch K-Means

def MiniKmean_Technique(ds):

    """

    Ref: http://scikit-learn.org/stable/modules/clustering.html#mini-batch-k-means

    """

    mkm = cluster.MiniBatchKMeans(n_clusters=n_clusters)

    return mkm.fit_predict(ds)



#   4.2.4 Define the function for clustering thru Spectral

def Spectral_Technique(ds):

    """

    Ref: http://scikit-learn.org/stable/modules/clustering.html#spectral-clustering

    """

    spectral = cluster.SpectralClustering(n_clusters=n_clusters)

    return spectral.fit_predict(ds)



#   4.2.5 Define the function for clustering thru DBSCAN

def Dbscan_Technique(ds):

    """

    Ref: http://scikit-learn.org/stable/modules/clustering.html#dbscan

    """

    dbscan = cluster.DBSCAN(eps=eps)

    return dbscan.fit_predict(ds)



#   4.2.6 Define the function for clustering thru Affinity Propagation

def AffProp_Technique(ds):

    """

    Ref: http://scikit-learn.org/stable/modules/clustering.html#affinity-propagation

    """

    ap = cluster.AffinityPropagation(damping=damping, preference=preference)

    return ap.fit_predict(ds)



#   4.2.7 Define the function for clustering thru Birch

def Birch_Technique(ds):

    """

    Ref: http://scikit-learn.org/stable/modules/clustering.html#birch

    """

    birch = cluster.Birch(n_clusters=n_clusters)

    return birch.fit_predict(ds)



#   4.2.8 Define the function for clustering thru Gaussian Mixture modeling

def Gmm_Technique(ds):

    """

    Ref: http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture

    """

    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')

    gmm.fit(ds)

    return gmm.predict(ds)



#   4.2.9 Function to evalauet the cluster performance using Silhouette Coefficient

#   Closer the value to 1, the better is the clustering.

def GetSilhouetteCoeff (ds, result):

    """

    Ref: http://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

    

    Input - 

        - ds - The dataset for which the clustering was done

        - result - The labels after the clustering

    """

    return metrics.silhouette_score(ds, result, metric=metric)



##  Unility method to extract the method name string after extracting it from DataFrame

#   This is temp method, unless a more elegant or standard solution is found

#   Need - Kmeans_Technique from DataFrame, instead getting

#   this - 0    Kmeans_Technique\nName: FunctionName, dtype: object    

def GetMethodName_Temp (method):

    m = str(method)

    m = m[1:m.index('\n')]

    return m.strip()
# Loop thru the various clustering techniques

for t in cluster_df.Technique:

    # Get the name of the method associated with the given technique

    method = cluster_df[cluster_df.Technique == t].FunctionName

    m = GetMethodName_Temp(method)    

    #print("Method", m)

    

    # Invoke the method to get the label for each record

    # Ref: https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-from-a-string-with-the-functions-name

    result = locals()[m](whr_data_for_clus)

    

    # Capture the output label for each of the record

    whr_data[t] = pd.DataFrame(result)

    # Execute & capture the Silhouette Coefficient for the each of the clustering response

    if t != 'Affinity Propagation' :

       cluster_df.loc[cluster_df.Technique == t, 'Silhouette.Coeff'] =  GetSilhouetteCoeff(whr_data_for_clus, result)



cluster_df.loc[cluster_df.Technique == 'Affinity Propagation', 'SilCoeff'] = 0
#   6.1 Cluster Analysis Data

whr_data.iloc[:,[0,12,13,14,15,16,17,18,19]]
#   6.2 Clustering Performance Evaluation

#   Clustering seems to perform well when the number of clusters are 2 as per Silhouette Coefficient

#   This is based on the results from testing with multiple cluster values = 2, 3, 5, 8, 10

#   If time provides, will add the detailed table with the results from other cluster numbers

cluster_df.iloc[:,[1, 2]]
rows = 4    # No of rows for the plot

cols = 2    # No of columns for the plot



cdf = cluster_df['Technique']



# 4 X 2 plot

fig,ax = plt.subplots(rows,cols, figsize=(15, 10)) 

x = 0

y = 0

for i in cdf:

   ax[x,y].scatter(whr_data.iloc[:, 6], whr_data.iloc[:, 5],  c=whr_data.iloc[:, 12+(x*y)])

   # Set the title for each of the plot

   ax[x,y].set_title(i + " Cluster Result")

    

   y = y + 1

   if y == cols:

       x = x + 1

       y = 0



plt.subplots_adjust(bottom=-0.5, top=1.5)

plt.show()

x = 0

y = 0
data = dict(type = 'choropleth', 

           locations = whr_data['Country'],

           locationmode = 'country names',

           z = whr_data['Happiness.Score'], 

           text = whr_data['Country'],

           colorbar = {'title':'Happiness Score'})

layout = dict(title = 'Global Happiness Score', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3) 
data = dict(type = 'choropleth', 

           locations = whr_data['Country'],

           locationmode = 'country names',

           z = whr_data['K-means'], 

           text = whr_data['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'K-Means Clustering Visualization', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)