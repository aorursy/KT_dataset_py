%reset -f

import time                   # To time processes

import warnings               # To suppress warnings



import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulatio 

import matplotlib.pyplot as plt                   # For graphics



from sklearn import cluster, mixture              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset



import os                     # For os related operations

import sys 



from sklearn import metrics



import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

# %matplotlib inline

warnings.filterwarnings('ignore')



import seaborn as sns
X= pd.read_csv("../input/2017.csv", header = 0)
X.columns.values
X.shape
X.dtypes
X_copy = X

X = X.iloc[:, 2: ]
## Function for various clusters

def compute_cluster (clType,df= X):

    if clType=='KMeans':

        result = cluster.KMeans(n_clusters= 2).fit_predict(df) 

    elif clType == 'MeanShift' :

        result = cluster.MeanShift(bandwidth=0.2).fit_predict(df)    

    elif clType == 'MiniBatchKMeans':

        result = cluster.MiniBatchKMeans(n_clusters=2).fit_predict(df)        

    elif clType == 'SpectralClustering':

        result = cluster.SpectralClustering(n_clusters=2).fit_predict(df)

    elif clType == 'DBSCAN':

        result = cluster.DBSCAN(eps=0.3).fit_predict(df)

    elif clType == 'Affinity Propagation':

        result = cluster.AffinityPropagation(damping=0.9, preference=-200).fit_predict(df)

    elif clType == 'Birch':

        result = cluster.Birch(n_clusters= 2).fit_predict(df)

    elif clType == 'GaussianMixture' :

        gmm = mixture.GaussianMixture( n_components=2, covariance_type='full')

        gmm.fit(df)

        result = gmm.predict(df)  

    else:

        print("exit")

    

    cl_df.loc[cl_df.Name == clType, 'Silhouette-Coeff'] = metrics.silhouette_score(df, result, metric='euclidean')

    cl_df.loc[cl_df.Name == clType, 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(df, result)

    

    return result
ss = StandardScaler().fit_transform(X)
cl_dist = {'Name' : ['KMeans','MeanShift','MiniBatchKMeans','SpectralClustering','DBSCAN','Affinity Propagation','Birch','GaussianMixture']}

cl_df = pd.DataFrame(cl_dist)

cl=pd.Series(['KMeans','MeanShift','MiniBatchKMeans','SpectralClustering','DBSCAN','Affinity Propagation','Birch','GaussianMixture'])
X.head(5)
for i in range(0,cl.size) :

    result = compute_cluster(clType=cl[i])

    X[cl[i]] = pd.DataFrame(result)
cl_df
rows = 4    # No of rows for the plot

cols = 2    # No of columns for the plot



# 4 X 2 plot

fig,ax = plt.subplots(rows,cols, figsize=(10, 10)) 

x = 0

y = 0

for i in cl:

    ax[x,y].scatter(X.iloc[:, 6], X.iloc[:, 5],  c=X.iloc[:, 12+(x*y)])

    ax[x,y].set_title(i + " Cluster Result")

    y = y + 1

    if y == cols:

        x = x + 1

        y = 0

        plt.subplots_adjust(bottom=-0.5, top=1.5)

plt.show()