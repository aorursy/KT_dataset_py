## Call libraries

import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulatio 

import matplotlib.pyplot as plt                   # For graphics



from sklearn import cluster, mixture              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset



import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) #additional initialization step to plot offline in Jupyter Notebooks
def explore_dataset(x):

    ds = x

    print("\nData set Attributes:\n")

    print("\nShape:\n",ds.shape)

    print("\nColumns:\n",ds.columns.values)

    print("\n1st 2 rows:\n",ds.head(2))

    print("\nData type:\n",ds.dtypes)

    #print("\nDataset info:\n",ds.info)

    print("\nDataset summary:\n",ds.describe())
x= pd.read_csv("../input/2017.csv", header = 0)

explore_dataset(x)

 
x= x.iloc[:, 2: ] 
# Instantiate scaler object

ss = StandardScaler()

# Use ot now to 'fit' &  'transform'

ss.fit_transform(x)
#Number of clusters

n_clusters = 2
# Instantiate object

km = cluster.KMeans(n_clusters =n_clusters )
# Fit the object to perform clustering

km_result = km.fit_predict(x)
# Draw scatter plot of two features, coloyued by clusters

plt.subplot(4, 2, 1)
plt.scatter(x.iloc[:, 4], x.iloc[:, 5],  c=km_result)
bandwidth = 0.1  
# No of clusters are NOT predecided

ms = cluster.MeanShift(bandwidth=bandwidth)
ms_result = ms.fit_predict(x)
plt.subplot(4, 2, 2)

plt.scatter(x.iloc[:, 4], x.iloc[:, 5],  c=ms_result)
two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)

two_means_result = two_means.fit_predict(x)

plt.subplot(4, 2, 3)

plt.scatter(x.iloc[:, 4], x.iloc[:, 5],  c= two_means_result)
spectral = cluster.SpectralClustering(n_clusters=n_clusters)

sp_result= spectral.fit_predict(x)

plt.subplot(4, 2, 4)



plt.scatter(x.iloc[:, 4], x.iloc[:, 5],  c=sp_result)
damping = 0.9

preference = -200



# 10.1  No of clusters are NOT predecided

affinity_propagation = cluster.AffinityPropagation(

        damping=damping, preference=preference)



#

affinity_propagation.fit(x)



# 

ap_result = affinity_propagation .predict(x)



# 

plt.subplot(4, 2, 6)

plt.scatter(x.iloc[:, 4], x.iloc[:, 5],  c=ap_result)
whdata = x
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

whdata.insert(0,'Country',x.iloc[:,0])

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