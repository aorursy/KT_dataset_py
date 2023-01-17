import numpy as np 

import os,time

import pandas as pd

from sklearn.manifold import TSNE

#!conda install tsnecuda -y -c cannylab

#from tsnecuda import TSNE

from sklearn.decomposition import PCA

%matplotlib inline

import matplotlib.pyplot as plt

!sed -i 's/from\ pandas.lib\ import\ Timestamp/from\ pandas\ import\ Timestamp/g;' /opt/conda/lib/python3.6/site-packages/ggplot/stats/smoothers.py

from ggplot import *



def dim_reduce(data,initial_dims = None,final_dims=2,use_pca=True):

    initial_dims = data.shape[1] if initial_dims is None else initial_dims

    print('Input data dimensions:',data.shape)

    if use_pca and initial_dims>50:

        pca = PCA(n_components=50)

        pca_result = pca.fit_transform(data)

        print('PCA output shape',pca_result.shape)

        data = pca_result

    time_start = time.time()

    tsne = TSNE(n_components=final_dims, verbose=1, perplexity=30, n_iter=1000)

    tsne_results = tsne.fit_transform(data)

    print('t-SNE done! Time elapsed: %r seconds'%(time.time()-time_start))

    print('t-SNE output shape:',tsne_results.shape)

    return tsne_results



def _2dplot(X_y_tuple,class_vector,hue_vector=None,xlabel='x-tsne',ylabel='y-tsne',hue_label='hue'):

    embeddings = {}

    hue_vector =  class_vector if hue_vector is None else hue_vector

    embeddings[xlabel],embeddings[ylabel] = X_y_tuple

    colours = ['red','blue','green','black','brown','violet','yellow','magenta','orange','purple']

    embeddings['label'] = [ colours[value] for value in class_vector]

    embeddings[hue_label] = hue_vector

    embeddings  = pd.DataFrame.from_dict(embeddings)

    chart = ggplot( embeddings, aes(x=xlabel, y=ylabel, color='label',alpha=hue_label))+ geom_point(size=70)+ ggtitle("tSNE dimensions colored by digit")

    return chart



# embeddings = dim_reduce(X,final_dims=2)

#_2dplot((embeddings[:,0],embeddings[:,1]),y_for_colors,y)
'#####################################################################################'
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0

length = 500

y = y_train[:length]

print(x_train.shape, y.shape)

X=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])[:length]

print(X.shape)
embeddings = dim_reduce(X,final_dims=2)

_2dplot((embeddings[:,0],embeddings[:,1]),y)