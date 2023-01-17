# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_origin = pd.read_csv('../input/fashion-mnist_train.csv', header = None)
df_labels = df_origin.iloc[1:, 0]
df_images = df_origin.iloc[1:, 1:]

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
         'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(df_images, df_labels, test_size=0.2)
print(X_train.shape, X_dev.shape, y_train.shape, y_dev.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

X_train_standardized = StandardScaler().fit_transform(X_train)
X_dev_standardized = StandardScaler().fit_transform(X_dev)
X_train_normalized = Normalizer().fit_transform(X_train_standardized)
X_dev_normalized = Normalizer().fit_transform(X_dev_standardized)
from sklearn.manifold import TSNE
#X_train_embeded = TSNE(n_components=2).fit_transform(X_train_normalized)
X_dev_tsne = TSNE(n_components=2).fit_transform(X_dev_normalized)
#Plot images of the fashion-mnist
num = 10 
plt.imshow(X_dev.values.astype(np.int)[num].reshape(28,28), cmap=plt.cm.binary)
# Visualize the embedding vectors

import numpy as np
from ast import literal_eval

colors = ['rgb(0,31,63)', 'rgb(255,133,27)', 'rgb(255,65,54)', 'rgb(0,116,217)', 'rgb(133,20,75)', 'rgb(57,204,204)',
'rgb(240,18,190)', 'rgb(46,204,64)', 'rgb(1,255,112)', 'rgb(255,220,0)',
'rgb(76,114,176)', 'rgb(85,168,104)', 'rgb(129,114,178)', 'rgb(100,181,205)']

def plot_embedding_v1(X_embeded, y):
    plt.rcParams["figure.figsize"] = [21, 18]
    for k, i in enumerate(np.unique(y.astype(np.int))):
        plt.scatter(X_embeded[y == i, 0],
                   X_embeded[y == i, 1],
                   color = '#%02x%02x%02x' % literal_eval(colors[k][3:]), 
                    label = labels[k])
    plt.legend()
    plt.show()

plot_embedding_v1(X_dev_tsne, y_dev)
# Visualize the embedding vectors
from matplotlib import offsetbox

def plot_embedding_v2(X, X_origin, title=None, dims=[None, 28, 28]):
    dims[0] = X.shape[0]
    X_origin = X_origin.values.astype(np.float).reshape(dims)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y_dev.values[i]),
                 color=plt.cm.Set1(y_dev.values.astype(np.int)[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 3e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X_origin[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


plot_embedding_v2(X_dev_tsne, X_dev, "t-SNE embedding of the digits)")
from sklearn import random_projection
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X_dev_normalized)
plot_embedding_v2(X_projected, X_dev, "Random Projection of the digits")
from time import time
from sklearn import decomposition
print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X_dev_normalized)
plot_embedding_v2(X_pca, X_dev,
               "Principal Components projection of the digits (time %.2fs)" %
               (time() - t0))
# Projection on to the first 2 linear discriminant components
from sklearn import discriminant_analysis

print("Computing Linear Discriminant Analysis projection")
X2 = X_dev_normalized.copy()
X2.flat[::X_dev_normalized.shape[1] + 1] += 0.01  # Make X invertible
t0 = time()
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)\
.fit_transform(X2, y_dev.values.astype(np.float))
plot_embedding_v2(X_lda, X_dev,
               "Linear Discriminant projection of the digits (time %.2fs)" %
               (time() - t0))
# Isomap projection of the digits dataset
from sklearn import manifold
print("Computing Isomap embedding")
n_neighbors = 30
t0 = time()
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X_dev_normalized)
print("Done.")
plot_embedding_v2(X_iso, X_dev,
               "Isomap projection of the digits (time %.2fs)" %
               (time() - t0))
# Locally linear embedding of the fashion-mnist dataset
print("Computing LLE embedding")
n_neighbors = 30
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
t0 = time()
X_lle = clf.fit_transform(X_dev_normalized)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding_v2(X_lle, X_dev,
               "Locally Linear Embedding of the digits (time %.2fs)" %
               (time() - t0))
# Modified Locally linear embedding
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='modified')
t0 = time()
X_mlle = clf.fit_transform(X_dev_normalized)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding_v2(X_mlle, X_dev,
               "Modified Locally Linear Embedding of the digits (time %.2fs)" %
               (time() - t0))
# MDS  embedding of the digits dataset
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
X_mds = clf.fit_transform(X_dev_normalized)
print("Done. Stress: %f" % clf.stress_)
plot_embedding_v2(X_mds, X_dev,
               "MDS embedding of the digits (time %.2fs)" %
               (time() - t0))
# Random Trees embedding of the digits dataset
from sklearn import ensemble
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
t0 = time()
X_transformed = hasher.fit_transform(X_dev_normalized)
pca = decomposition.TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_transformed)

plot_embedding_v2(X_reduced, X_dev,
               "Random forest embedding of the digits (time %.2fs)" %
               (time() - t0))

print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
t0 = time()
X_se = embedder.fit_transform(X_dev_normalized)

plot_embedding_v2(X_se, X_dev,
               "Spectral embedding of the digits (time %.2fs)" %
               (time() - t0))
