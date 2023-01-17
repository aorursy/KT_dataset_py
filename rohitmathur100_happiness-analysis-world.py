import time                   # To time processes

import warnings              

import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulatio 

import matplotlib.pyplot as plt                   # For graphics



from sklearn import cluster, mixture              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset



import os                     # For os related operations

import sys                    # For data 



import plotly

from plotly.graph_objs import Scatter, Layout

import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

plotly.offline.init_notebook_mode(connected=True)

%matplotlib inline

warnings.filterwarnings('ignore')

#os.chdir("C:\\Big Data R\\python\\Happiness")

X= pd.read_csv("../input/2017.csv", header = 0)

X_tr = X

Y_tr= X

Z_tr = X

A_tr=X

B_tr =X

G_tr=X

X.columns.values

X.shape                 # 155 X 12

X = X.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns

X.head(2)


#######

ss = StandardScaler()

# 3.1.3 Use ot now to 'fit' &  'transform'

ss.fit_transform(X)







#### 4. Begin Clustering   

                                  

# 5.1 How many clusters

#     NOT all algorithms require this parameter

n_clusters = 4    

km = cluster.KMeans(n_clusters =n_clusters )



# 5.2.1 Fit the object to perform clustering

km_result = km.fit_predict(X)



# 5.3 Draw scatter plot of two features, coloyued by clusters

plt.subplot(4, 2, 1)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=km_result)
## 9. DBSCAN

# http://scikit-learn.org/stable/modules/clustering.html#dbscan

#   The DBSCAN algorithm views clusters as areas of high density separated

#    by areas of low density. Due to this rather generic view, clusters found

#     by DBSCAN can be any shape, as opposed to k-means which assumes that

#      clusters are convex shaped.    

#    Parameter eps decides the incremental search area within which density

#     should be same



eps = 0.3

# 9.1 No of clusters are NOT predecided

dbscan = cluster.DBSCAN(eps=eps)

# 9.2

db_result= dbscan.fit_predict(X)

# 9.3

plt.subplot(4, 2, 5)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5], c=db_result)
labels=db_result

Z_tr['clusters']=labels



# Create a pandas data-frame of country vs cluster labelYX_tr = X_tr.iloc[:,[0,12]]



data = [dict(type = 'choropleth', 

                locations = Z_tr['Country'],

                locationmode = 'country names',

                z = Z_tr['clusters'], 

                   text = Z_tr['Country'],

                  colorbar = {'title':'Happiness'})

        ]

layout = dict(title = 'Global Happiness Using  DB Clustering',

                     geo = dict(showframe = False, 

                               projection = {'type': 'Mercator'})

             )



choromap3 = go.Figure(data = data, layout=layout)



iplot(choromap3)
# 10. Affinity Propagation

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



damping = 0.9

preference = -200



# 10.1  No of clusters are NOT predecided

affinity_propagation = cluster.AffinityPropagation(

        damping=damping, preference=preference)



# 10.2

affinity_propagation.fit(X)



# 10.3

ap_result = affinity_propagation .predict(X)



# 10.4

plt.subplot(4, 2, 6)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=ap_result)
labels=ap_result  #Add the column into our list

A_tr['clusters']=labels



# Create a pandas data-frame of country vs cluster labelYX_tr = X_tr.iloc[:,[0,12]]



data = [dict(type = 'choropleth', 

                locations = A_tr['Country'],

                locationmode = 'country names',

                z = A_tr['clusters'], 

                   text = A_tr['Country'],

                  colorbar = {'title':'Happiness'})

        ]

layout = dict(title = 'Global Happiness Using  Affinity Propogation Clustering',

                     geo = dict(showframe = False, 

                               projection = {'type': 'Mercator'})

             )



choromap3 = go.Figure(data = data, layout=layout)



iplot(choromap3,validate=False)
# 14. Gaussian Mixture modeling

#  http://203.122.28.230/moodle/course/view.php?id=6&sectionid=11#section-3

#  It treats each dense region as if produced by a gaussian process and then

#  goes about to find the parameters of the process



# 14.1

gmm = mixture.GaussianMixture( n_components=n_clusters, covariance_type='full')



# 14.2

gmm.fit(X)



# 14.3

gmm_result = gmm.predict(X)

plt.subplot(4, 2, 8)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=gmm_result)
#########################################################



labels=gmm_result  #Add the column into our list

G_tr['clusters']=labels



# Create a pandas data-frame of country vs cluster labelYX_tr = X_tr.iloc[:,[0,12]]



data = [dict(type = 'choropleth', 

                locations = G_tr['Country'],

                locationmode = 'country names',

                z = G_tr['clusters'], 

                   text = G_tr['Country'],

                  colorbar = {'title':'Happiness'})

        ]

layout = dict(title = 'Global Happiness Using  Gausain Mixture Model Clustering',

                     geo = dict(showframe = False, 

                               projection = {'type': 'Mercator'})

             )



choromap3 = go.Figure(data = data, layout=layout)



iplot(choromap3,validate=False)
## 11. Birch

# http://scikit-learn.org/stable/modules/clustering.html#birch    

# The Birch builds a tree called the Characteristic Feature Tree (CFT) for the

#   given data and clustering is performed as per the nodes of the tree



# 11.1

birch = cluster.Birch(n_clusters=n_clusters)



# 11.2

birch_result = birch.fit_predict(X)



# 11.3

plt.subplot(4, 2, 7)

plt.scatter(X.iloc[:, 4], X.iloc[:, 5],  c=birch_result)
labels=birch_result  #Add the column into our list

B_tr['clusters']=labels



# Create a pandas data-frame of country vs cluster labelYX_tr = X_tr.iloc[:,[0,12]]



data = [dict(type = 'choropleth', 

                locations = B_tr['Country'],

                locationmode = 'country names',

                z = B_tr['clusters'], 

                   text = B_tr['Country'],

                  colorbar = {'title':'Happiness'})

        ]

layout = dict(title = 'Global Happiness Using  Birch Clustering',

                     geo = dict(showframe = False, 

                               projection = {'type': 'Mercator'})

             )



choromap3 = go.Figure(data = data, layout=layout)



iplot(choromap3,validate=False)