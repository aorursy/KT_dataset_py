## Call libraries

import numpy as np            # Data manipulation

import pandas as pd           # Dataframe manipulatio 

import matplotlib.pyplot as plt                   # For graphics



from sklearn import cluster, mixture              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset



import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) #additional initialization step to plot offline in Jupyter Notebooks
def explore_dataset(data):

    ds = data

    print("\nData set Attributes:\n")

    print("\nShape:\n",ds.shape)

    print("\nColumns:\n",ds.columns.values)

    print("\n1st 2 rows:\n",ds.head(2))

    print("\nData type:\n",ds.dtypes)

    #print("\nDataset info:\n",ds.info)

    print("\nDataset summary:\n",ds.describe())
# Function to normalize and transform dataset for easier parameter selection



def scaler_obj(data):

    ds=data

    ss = StandardScaler()

    ss.fit_transform(ds) # 'fit' &  'transform'

    return(ds)
# Function to accept and implement various clustering algorithms



def cluster_model(data, model_name, input_param):

    ds = data

    params = input_param

    if str.lower(model_name) == 'kmeans':                                ## KMeans

        cluster_obj = cluster.KMeans(n_clusters=params['n_clusters'])

    if str.lower(model_name) == str.lower('MiniBatchKMeans'):            ## Mini Batch K-Means

        cluster_obj = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])

    if str.lower(model_name) == str.lower('SpectralClustering'):         ## Spectral clustering

        cluster_obj = cluster.SpectralClustering(n_clusters=params['n_clusters'])

    if str.lower(model_name) == str.lower('MeanShift'):                  ## Mean Shift

        cluster_obj = cluster.MeanShift(bandwidth=params['bandwidth'])

    if str.lower(model_name) == str.lower('DBSCAN'):                     ## DBSCAN

        cluster_obj = cluster.DBSCAN(eps=params['eps'])

    if str.lower(model_name) == str.lower('AffinityPropagation'):        ## Affinity Propagation

        cluster_obj = cluster.AffinityPropagation(damping=params['damping'], 

                                                  preference=params['preference'])

        cluster_obj.fit(ds)

    if str.lower(model_name) == str.lower('Birch'):                      ## Birch

        cluster_obj = cluster.Birch(n_clusters=input_param['n_clusters'])

    if str.lower(model_name) == str.lower('GaussianMixture'):            ## Gaussian Mixture modeling

        cluster_obj = mixture.GaussianMixture(n_components=params['n_clusters'], 

                                              covariance_type='full')

        cluster_obj.fit(ds)

    

    ### Fit the object to perform clustering

    if str.lower(model_name) in ['affinitypropagation', 'gaussianmixture']:

        model_result = cluster_obj.predict(ds)

    else:

        model_result = cluster_obj.fit_predict(ds)

    

    ### Store the results obtained from the clustering algorithm

    ds[model_name] = pd.DataFrame(model_result)

    

    return(model_result)
## Function for scatter plot

def scatter_plot(model_name, n_row, n_col, position, data, model_result):

    ds=data

    algo_result = model_result

    plt.subplot(n_row, n_col, position)

    plt.title(model_name)

    plt.scatter(ds.iloc[:, 4], ds.iloc[:, 5],  c=algo_result)

    plt.subplots_adjust(bottom=0.03, top=0.97, hspace=0.25)

    return()
# Read dataset

gcds= pd.read_csv("../input/2017.csv", header = 0)



# Explore dataset

explore_dataset(gcds)
gcds_h = gcds.iloc[:, 2: ]      # Ignore Country and Happiness_Rank columns
gcds_h = scaler_obj(gcds_h)
#Initialize variables for figure plotting

n_row = 4

n_col = 2

#Initialize list of models and respective parameters

cluster_list = ["KMeans", "MiniBatchKMeans", "SpectralClustering", "MeanShift",

                "DBSCAN", "Birch", "GaussianMixture", "AffinityPropagation"]

input_param = {'n_clusters':2, 'bandwidth':0.1, "damping":0.9, "eps":0.3,

               "preference":-200}



plt.figure(figsize=(15,15))

position = 1
#Traverse through the list of clusters and plot the results

for i in cluster_list:

    cluster_result = cluster_model(gcds_h,i,input_param)

    scatter_plot(i,n_row, n_col, position, gcds_h, cluster_result)

    position = position + 1

plt.show()
# make a copy of the dataset

gcds_map = gcds_h

# Add the country column from the intial dataset

gcds_map.insert(0,'Country',gcds['Country'])
gcds_map.iloc[:5,11:]
# Prepare parameters for the mapping

def map_plot(data_set,col_name):

    ds=data_set

    data = [dict(

        type = 'choropleth',

        locations = ds['Country'],

        locationmode = 'country names',

        z = ds[col_name],

        text = ds['Country'],

        colorbar = dict(

            title = 'Cluster Group'

        )

      )]



    layout = dict(

        title = col_name,

        geo = dict(

            showframe = False,

            projection = dict(

                type = 'Mercator'

            )

        )

    )

    # Map plotting

    choromap = go.Figure( data=data, layout=layout )

    iplot(choromap)

    return()
map_plot(gcds_map,'Happiness.Score')
for j in cluster_list:

    map_plot(gcds_map,j)