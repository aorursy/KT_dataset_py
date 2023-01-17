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
#os.chdir("E:/Big_Data/Code/Python/exercise/Clustering_Algorithms")

WHR_data = pd.read_csv("../input/2017.csv", header = 0)



#To play with "World Happiness Report" Data set, we will create a copy 

WHR_data_copy = WHR_data.copy(deep = True)



# preview Data

print(WHR_data.info())

WHR_data.shape

WHR_data.head()

WHR_data.sample(15)



WHR_data.columns

plt.figure(figsize=(12,8))

sns.heatmap(WHR_data.corr())


sns.pairplot(WHR_data[['Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.high','Whisker.low', 'Economy..GDP.per.Capita.', 'Family']]);

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# For offline use

import cufflinks as cf



cf.go_offline()

WHR_data[['Happiness.Score','Whisker.high','Family','Freedom','Dystopia.Residual']].iplot(kind='spread')




#Ignore Country and Happiness_Rank Columns



WHR_data = WHR_data.iloc[:,2:]



print("\n \n Dimenstion of dataset  : WHR_data.shape")

WHR_data.shape



WHR_data.dtypes

# Instantiate Scaler Object 

ss = StandardScaler()

# Fit and transform 

ss.fit_transform(WHR_data)

WHR_data.sample(20)


# Define CluserMethod class : which returns the clustering result based on input 



class ClusterMethodList(object) :

    def get_cluster_instance(self, argument,input_data,X):

        method_name = str(argument).lower()+ '_cluster'

        method = getattr(self,method_name,lambda : "Invalid Clustering method")

        return method(input_data,X)

    

    def kmeans_cluster(self,input_data,X):

        km = cluster.KMeans(n_clusters =input_data['n_clusters'] )

        return km.fit_predict(X)

   

    def meanshift_cluster(self,input_data,X):

        ms = cluster.MeanShift(bandwidth=input_data['bandwidth'])

        return  ms.fit_predict(X)

    

    def minibatchkmeans_cluster(self,input_data,X):

        two_means = cluster.MiniBatchKMeans(n_clusters=input_data['n_clusters'])

        return two_means.fit_predict(X)

   

    def dbscan_cluster(self,input_data,X):

        db = cluster.DBSCAN(eps=input_data['eps'])

        return db.fit_predict(X)

    

    def spectral_cluster(self,input_data,X):

        sp = cluster.SpectralClustering(n_clusters=input_data['n_clusters'])

        return sp.fit_predict(X)

   

    def affinitypropagation_cluster(self,input_data,X):

        affinity_propagation =  cluster.AffinityPropagation(damping=input_data['damping'], preference=input_data['preference'])

        affinity_propagation.fit(X)

        return affinity_propagation.predict(X)

       

    

    def birch_cluster(self,input_data,X):

        birch = cluster.Birch(n_clusters=input_data['n_clusters'])

        return birch.fit_predict(X)

   

    def gaussian_mixture_cluster(self,input_data,X):

        gmm = mixture.GaussianMixture( n_components=input_data['n_clusters'], covariance_type='full')

        gmm.fit(X)

        return  gmm.predict(X)



# Define Clustering Prcoess



def startClusteringProcess(list_cluster_method,input_data,no_columns,data_set):

    fig,ax = plt.subplots(no_rows,no_columns, figsize=(10,10)) 

    cluster_list = ClusterMethodList()

    i = 0

    j=0

    for cl in list_cluster_method :

        cluster_result = cluster_list.get_cluster_instance(cl,input_data,data_set)

        #convert cluster result array to DataFrame

        data_set[cl] = pd.DataFrame(cluster_result)

        ax[i,j].scatter(data_set.iloc[:, 4], data_set.iloc[:, 5],  c=cluster_result)

        ax[i,j].set_title(cl+" Cluster Result")

        j=j+1

        if( j % no_columns == 0) :

            j= 0

            i=i+1

    plt.subplots_adjust(bottom=-0.5, top=1.5)

    plt.show()



list_cluster_method = ['KMeans',"MeanShift","MiniBatchKmeans","DBScan","Spectral","AffinityPropagation","Birch","Gaussian_Mixture"]

# For Graph display 

no_columns = 2

no_rows = 4

# NOT all algorithms require this parameter

n_clusters= 3

bandwidth = 0.1 

# eps for DBSCAN

eps = 0.3

## Damping and perference for Affinity Propagation clustering method

damping = 0.9

preference = -200

input_data = {'n_clusters' :  n_clusters, 'eps' : eps,'bandwidth' : bandwidth, 'damping' : damping, 'preference' : preference}

# Start Clustering Process

startClusteringProcess(list_cluster_method,input_data,no_columns,WHR_data)

WHR_data.insert(0,'Country',WHR_data_copy.iloc[:,0])

WHR_data.iloc[:,[0,11,12,13,14,15,16,17,18]]




data = dict(type = 'choropleth', 

           locations = WHR_data['Country'],

           locationmode = 'country names',

           z = WHR_data['Gaussian_Mixture'], 

           text = WHR_data['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Gaussian Mixture Clustering Visualization', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)



data = dict(type = 'choropleth', 

           locations = WHR_data['Country'],

           locationmode = 'country names',

           z = WHR_data['KMeans'], 

           text = WHR_data['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'K-Means Clustering Visualization', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)



data = dict(type = 'choropleth', 

           locations = WHR_data['Country'],

           locationmode = 'country names',

           z = WHR_data['Happiness.Score'], 

           text = WHR_data['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'Global Happiness Score', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
