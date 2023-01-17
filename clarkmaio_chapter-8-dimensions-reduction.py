import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.manifold import TSNE

from sklearn.datasets import load_digits



import umap



import time

import tensorflow as tf

def plot_scatter_class(df, x, y, c, title = ''):

    

    #cmap = {0:'#1f77b4', 1: '#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 6:'#e377c2', 7:'#7f7f7f', 8:'#bcbd22', 9:'#17becf'}

    

    plt.figure(figsize = (15, 15))

    plt.scatter(df.loc[:, x], df.loc[:, y], c = c)

    plt.grid()

    plt.title(title)

    plt.legend()

    plt.xlabel(str(x))

    plt.ylabel(str(y))
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_reshape = x_train.reshape(x_train.shape[0], -1)

x_test_reshape = x_test.reshape(x_test.shape[0], -1)



print ('New shape: {}'.format(x_train_reshape.shape))
# Model

rf = RandomForestClassifier()
# No reduction

t = time.time()

rf.fit(x_train_reshape, y_train)

T = time.time()-t

y_pred = rf.predict(x_test_reshape)



print('Training time: {}'.format(T))

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
# Reduction analysis

pca_analysis = PCA(svd_solver = 'full', n_components = x_train_reshape.shape[1])

pca_analysis.fit(x_train_reshape)

variance_cumsum = np.cumsum(pca_analysis.explained_variance_ratio_)



plt.figure(figsize = (20, 10))

plt.plot(range(1, len(variance_cumsum)+1), variance_cumsum)

plt.grid()

plt.title('Variance cumsum')

plt.xlabel('Components')

plt.ylabel('Explained variance')
# PCA reduction

var_reduction = .95

pca = PCA(n_components = var_reduction)



x_train_pca = pca.fit_transform(x_train_reshape)

x_test_pca = pca.transform(x_test_reshape)



t = time.time()

rf.fit(x_train_pca, y_train)

T = time.time()-t

y_pred = rf.predict(x_test_pca)



print('Training time: {}'.format(T))

print('Variance reduction: {}. Nmber of components: {}.'.format(var_reduction, x_train_pca.shape[1]))

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
"""

# t-SNE

tsne = TSNE(n_components = 2)

x_tsne = tsne.fit_transform(x_train_pca)

df_tsne = pd.DataFrame(x_tsne)



plot_scatter_class(df_tsne, 0, 1, y_train, title = 't SNE')

"""
# UMAP

umap_mdl = umap.UMAP(n_components = 2)

x_train_umap = umap_mdl.fit_transform(x_train_pca)

x_test_umap = umap_mdl.transform(x_test_pca)

df_train_umap = pd.DataFrame(x_train_umap)

df_test_umap = pd.DataFrame(x_test_umap)
plot_scatter_class(df_train_umap, 0, 1, y_train, title = 'UMAP TRAIN SET')
plot_scatter_class(df_test_umap, 0, 1, y_test, title = 'UMAP TEST SET')
# UMAP reduction

t = time.time()

rf.fit(x_train_umap, y_train)

T = time.time()-t

y_pred = rf.predict(x_test_umap)



print('Training time: {}'.format(T))

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))