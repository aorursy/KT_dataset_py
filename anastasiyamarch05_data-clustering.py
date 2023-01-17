import numpy as np

import pandas as pd

import seaborn as sns

import sklearn

from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA as PCA

from sklearn.metrics import silhouette_samples

import matplotlib.pyplot as plt

from matplotlib import cm

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/ammonium-prediction-in-river-water/test.csv')
print('shape = ', data.shape)

print(data.columns)
data = data.drop(['Id',], axis=1)

data = data.dropna()

data.info()



data.head(10)
g = sns.PairGrid(data[['1', '2', '3', '4']])

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=10);
g = sns.PairGrid(data[['1', '2', '3', '4']], palette="GnBu_d")

g.map(plt.scatter, s=50, edgecolor="white")

g.add_legend();
g = sns.PairGrid(data[['1', '2', '3', '4']])

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();
g = sns.PairGrid(data[['1', '2', '3', '4']])

g = g.map_diag(plt.hist, histtype="step", linewidth=3)

g = g.map_offdiag(plt.scatter)

g = g.add_legend()
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3,3,1)

sns.barplot(y='1',x='2', data=data);

ax2 = fig.add_subplot(3,3,2)

sns.barplot(y='3',x='4', data=data);

ax3 = fig.add_subplot(3,3,3)

sns.barplot(y='5',x='6', data=data);


pca = PCA(n_components=2)

pca.fit(data)

Xpca = pca.transform(data)

sns.set()

plt.figure(figsize=(8,8))

plt.scatter(Xpca[:,0],Xpca[:,1], c='Red')

plt.show()
from sklearn.manifold import TSNE

tsn = TSNE()

res_tsne = tsn.fit_transform(data)

plt.figure(figsize=(8,8))

plt.scatter(res_tsne[:,0],res_tsne[:,1]);
from sklearn.cluster import AgglomerativeClustering as AggClus
clus_mod = AggClus(n_clusters=3)

assign = clus_mod.fit_predict(data)

plt.figure(figsize=(8,8))

sns.set(style='darkgrid',palette='muted')

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=assign, palette='Set1');
from scipy.cluster.hierarchy import dendrogram, ward

sns.set(style='white')

plt.figure(figsize=(10,7))

link = ward(res_tsne)

dendrogram(link)

ax = plt.gca()

bounds = ax.get_xbound()

ax.plot(bounds, [30,30],'--', c='k')

ax.plot(bounds,'--', c='k')

plt.show()
clus_mod = AggClus(n_clusters=5)

assign = clus_mod.fit_predict(data)

plt.figure(figsize=(8,8))

sns.set(style='darkgrid',palette='muted')

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=50, hue=assign, palette='copper');