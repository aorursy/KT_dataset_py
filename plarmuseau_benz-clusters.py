import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import time

import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.metrics.pairwise import pairwise_distances_argmin

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import Birch



# read datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



test['y'] = 102  # to make append possible

y_train = train["y"]

#find not unique ?

kolom=train.columns 

kolom=[k for k in kolom if k not in ['ID','y']]

train_u = train.sort_values(by='y').duplicated(subset=kolom)

#print(train[train_u==True])



# find not unique combinations X10...



kolom_p=train.columns 

kolom_p=[k for k in kolom if k not in ['ID','y','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9']]

#print(kolom_p)

#train_p = train.groupby(kolom_p).min()

#print()

#print(train_p)





# find not unique combinations lettercombinations.





kolom_t=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

#print(kolom_t)

train_t = train.groupby(kolom_t).min()

#print(train_t)

#print(train.ix[train['ID']==3580])





test['y'] = 102

totaal= train.append(test)

test = test.drop(['y'], axis=1)

train_t = train.groupby(kolom_t,as_index=False).min()



for c in totaal.columns:

    if totaal[c].dtype == 'object':

        tempt = totaal[['y',c]]

        temp=tempt.groupby(c).mean().sort('y')

        templ=temp.index

        print(templ)

        aant=len(templ)

        train_t[c].replace(to_replace=templ, value=[x/aant for x in range(0,aant)], inplace=True, method='pad', axis=1)

        test[c].replace(to_replace=templ, value=[x/aant for x in range(0,aant)], inplace=True, method='pad', axis=1)        



kolom_t=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

#print(kolom_t)





kolom_c=[k for k in kolom if k not in ['ID','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9']]

train_t['som']=train_t[kolom_c].sum(axis=1,skipna=True)

#print(train_t) 
kolom_c=[k for k in kolom if k not in ['ID','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9']]



kolom_cl=['X0','X1','X2','X3','X4','X5','X6','X8']

X=train_t.drop('ID',axis=1)



#kmeans

k_means = KMeans(init='k-means++', n_clusters=25, n_init=10)

k_means.fit(X)



#mbk

mbk = MiniBatchKMeans(init='k-means++', n_clusters=25, batch_size=45,

                      n_init=10, max_no_improvement=10, verbose=0)

mbk.fit(X)



#birch

brc = Birch(branching_factor=50, n_clusters=25, threshold=0.5, compute_labels=True)

brc.fit(X)



#spectral

spectral = cluster.SpectralClustering(n_clusters=25, eigen_solver='arpack', affinity="nearest_neighbors")

spectral.fit(X)



kolom_c=[k for k in kolom if k not in ['ID','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9']]



#finding labels

k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)

mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)

k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

order = pairwise_distances_argmin(k_means_cluster_centers,

                                  mbk_means_cluster_centers)

brc_labels=brc.predict(X)

spec_labels=spectral.labels_ 



#throwing everything in a plotdata

plotdata=train_t[['ID','y','X0','X5']]

plotdata['kml']=k_means_labels

plotdata['mbk']=mbk_means_labels

plotdata['brc']=brc_labels

plotdata['spect']=spec_labels

plotdata['som']=train_t[kolom_c].sum(axis=1,skipna=True)

plotdata=plotdata[plotdata['y']<150]

print(plotdata)





#graphing

sns.set(style="white")

g = sns.PairGrid(plotdata, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot, lw=3)