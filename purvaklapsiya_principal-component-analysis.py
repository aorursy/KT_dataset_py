# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Necessary imports

%matplotlib inline

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

plt.style.use('ggplot')



import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})



import plotly

# connected=True means it will download the latest version of plotly javascript library.

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import plotly.figure_factory as ff



import warnings

warnings.filterwarnings('ignore')





#!------------- Dimenionality reduction imports --------!#

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.manifold import LocallyLinearEmbedding

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.decomposition import KernelPCA

from sklearn.decomposition import SparsePCA

from sklearn.manifold import MDS

from sklearn.manifold import Isomap

from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import GaussianRandomProjection

from sklearn.decomposition import FastICA

from sklearn.decomposition import MiniBatchDictionaryLearning

from sklearn.random_projection import SparseRandomProjection

import keras

from keras.models import Sequential, Model

from keras.layers import Dense

from keras.optimizers import Adam
train = pd.read_csv('../input/train.csv')
## Covariance Matrix

def covariance_matric_equation(X_std):    

    mean_vec = np.mean(X_std, axis=0)

    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

    return cov_mat



def covariance_matrix_function(X_std):

    return np.cov(X_std.T)



#Eigen Decomposition

def eigen_decomposition(cov_mat):

    return np.linalg.eig(cov_mat)



def eigen_decomposition_square(cov_mat):

    #numerically stable for square matrix

    return np.linalg.eigh(cov_mat)



# Correlation Matrix

def correlation_matrix(X_std):

    return np.corrcoef(X_std)

def svd(X):

    

  # Data matrix X, X doesn't need to be 0-centered

  n, m = X.shape

  # Compute full SVD

  U, Sigma, Vh = np.linalg.svd(X, 

      full_matrices=False, # It's not necessary to compute the full matrix of U or V

      compute_uv=True)

  # Transform X with SVD components

  X_svd = (U[:, :5]*Sigma[:5]).round(2)

  return X_svd



def pca(X):

    n, m = X.shape

    assert np.allclose(X.mean(axis=0), np.zeros(m))

      # Compute covariance matrix

    C = covariance_matric_equation(X)

      # Eigen decomposition

    eigen_vals, eigen_vecs = eigen_decomposition(C)

    print("pca eigen vals:",eigen_vals)

    # Project X onto PC space

    X_pca = np.dot(X, eigen_vecs)

    return X_pca



from sklearn.preprocessing import StandardScaler

def standarized_data(X):

    return StandardScaler().fit_transform(X)
# save the labels to a Pandas series target

target = train['label']

# Drop the label feature

train = train.drop("label",axis=1)



X_std = standarized_data(train.values)

pca_x = pca(X_std)

svd_x = svd(X_std)
from sklearn.decomposition import PCA

from sklearn.decomposition import TruncatedSVD

svd_x = svd(X_std)

svd_x.shape
X= train[:6000].values

del train

# Standardising the values

X_std = StandardScaler().fit_transform(X)



# Call the PCA method with 5 components. 

pca = PCA(n_components=5)

pca.fit(X_std)

X_5d = pca.transform(X_std)



# For cluster coloring in our Plotly plots, remember to also restrict the target values 

Target = target[:6000]
svd_tr = TruncatedSVD(n_components=5)

svd_tr.fit(X_std)

Xs_5d = svd_tr.transform(X_std)
import plotly.offline as py

py.init_notebook_mode(connected=True)



trace0 = go.Scatter(

    x = X_5d[:,0],

    y = X_5d[:,1],

#     name = Target,

#     hoveron = Target,

    mode = 'markers',

    text = Target,

    showlegend = False,

    marker = dict(

        size = 8,

        color = Target,

        colorscale ='Jet',

        showscale = False,

        line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        ),

        opacity = 0.8

    )

)

data = [trace0]



layout = go.Layout(

    title= 'Principal Component Analysis (PCA)',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)





fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
import plotly.offline as py

py.init_notebook_mode(connected=True)



trace0 = go.Scatter(

    x = Xs_5d[:,1],

    y = Xs_5d[:,2],

#     name = Target,

#     hoveron = Target,

    mode = 'markers',

    text = Target,

    showlegend = False,

    marker = dict(

        size = 8,

        color = Target,

        colorscale ='Jet',

        showscale = False,

        line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        ),

        opacity = 0.8

    )

)

data = [trace0]



layout = go.Layout(

    title= 'Principal Component Analysis (PCA)',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)





fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
import plotly.offline as py

py.init_notebook_mode(connected=True)



trace0 = go.Scatter(

    x = svd_x[:,1],

    y = svd_x[:,2],

#     name = Target,

#     hoveron = Target,

    mode = 'markers',

    text = Target,

    showlegend = False,

    marker = dict(

        size = 8,

        color = Target,

        colorscale ='Jet',

        showscale = False,

        line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        ),

        opacity = 0.8

    )

)

data = [trace0]



layout = go.Layout(

    title= 'Principal Component Analysis (PCA)',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)





fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
from sklearn.cluster import KMeans # KMeans clustering 

# Set a KMeans clustering with 9 components ( 9 chosen sneakily ;) as hopefully we get back our 9 class labels)

kmeans = KMeans(n_clusters=9)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(X_5d)



trace_Kmeans = go.Scatter(x=X_5d[:, 0], y= X_5d[:, 1], mode="markers",

                    showlegend=False,

                    marker=dict(

                            size=8,

                            color = X_clustered,

                            colorscale = 'Portland',

                            showscale=False, 

                            line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        )

                   ))



layout = go.Layout(

    title= 'KMeans Clustering',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)



data = [trace_Kmeans]

fig1 = dict(data=data, layout= layout)

# fig1.append_trace(contour_list)

py.iplot(fig1, filename="svm")
from sklearn.cluster import KMeans # KMeans clustering 

# Set a KMeans clustering with 9 components ( 9 chosen sneakily ;) as hopefully we get back our 9 class labels)

kmeans = KMeans(n_clusters=9)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(X_5d)



trace_Kmeans = go.Scatter(x=Xs_5d[:, 0], y= Xs_5d[:, 1], mode="markers",

                    showlegend=False,

                    marker=dict(

                            size=8,

                            color = X_clustered,

                            colorscale = 'Portland',

                            showscale=False, 

                            line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        )

                   ))



layout = go.Layout(

    title= 'KMeans Clustering',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)



data = [trace_Kmeans]

fig1 = dict(data=data, layout= layout)

# fig1.append_trace(contour_list)

py.iplot(fig1, filename="svm")
from sklearn.cluster import KMeans # KMeans clustering 

# Set a KMeans clustering with 9 components ( 9 chosen sneakily ;) as hopefully we get back our 9 class labels)

kmeans = KMeans(n_clusters=9)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(X_5d)



trace_Kmeans = go.Scatter(x=svd_x[:, 0], y= svd_x[:, 1], mode="markers",

                    showlegend=False,

                    marker=dict(

                            size=8,

                            color = X_clustered,

                            colorscale = 'Portland',

                            showscale=False, 

                            line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        )

                   ))



layout = go.Layout(

    title= 'KMeans Clustering',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)



data = [trace_Kmeans]

fig1 = dict(data=data, layout= layout)

# fig1.append_trace(contour_list)

py.iplot(fig1, filename="svm")