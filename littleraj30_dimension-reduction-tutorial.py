import numpy as np 

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

print(os.listdir("../input"))
train=pd.read_csv("../input/fashion-mnist_train.csv")
train.head()
train_label=train['label']

del train['label']
# Batch data

train=train.iloc[0:1000,:]

train_label=train_label.iloc[0:1000]
# helper function for plotting

def plotting(xdf,ydf,title):

    xdf=pd.DataFrame(data=xdf, index=train.index)

    xdf=xdf.iloc[:,0:2]

    data=pd.concat((xdf,ydf),axis=1)

    data=data.rename(columns={0:'first',1:'second'})

    sns.lmplot(data=data,x='first',y='second',hue='label',fit_reg=False)

    sns.set_style('darkgrid')

    plt.title(title)
from sklearn.decomposition import PCA

pca=PCA(n_components=2,random_state=132,whiten=False)

train_pca=pca.fit_transform(train)

plotting(train_pca,train_label,'PCA')
from sklearn.decomposition import IncrementalPCA

ipca=IncrementalPCA(n_components=3)

batch_size=100

for train_batch in np.array_split(train,batch_size):

    ipca.partial_fit(train_batch)

plotting(ipca.transform(train),train_label,'Incremental PCA')
from sklearn.decomposition import SparsePCA

sparse_pca=SparsePCA(n_components=2,alpha=0.002)

plotting(sparse_pca.fit_transform(train),train_label,'Incremental PCA')
from sklearn.decomposition import KernelPCA

kpca=KernelPCA(n_components=2, kernel='rbf')

plotting(kpca.fit_transform(train),train_label,'Kernel PCA with RBF')
from sklearn.decomposition import TruncatedSVD

tsvd=TruncatedSVD(n_components=2, n_iter=5, algorithm='randomized')

plotting(tsvd.fit_transform(train),train_label,'SVD')
from sklearn.random_projection import GaussianRandomProjection

grp=GaussianRandomProjection(n_components='auto', eps=0.5)

plotting(grp.fit_transform(train),train_label,'Gaussian Random Projection')
from sklearn.random_projection import SparseRandomProjection

srp=SparseRandomProjection(n_components='auto', eps=0.5, dense_output=False)

plotting(srp.fit_transform(train),train_label,'Gaussian Random Projection')
from sklearn.manifold import Isomap

isomap= Isomap(n_components=2, n_neighbors=10, n_jobs=-1)

plotting(isomap.fit_transform(train),train_label,'ISOMAP')
from sklearn.manifold import TSNE

tsne=TSNE(n_components=2, learning_rate=300, early_exaggeration=12, init='random')

plotting(tsne.fit_transform(train),train_label,'TSNE')
from sklearn.manifold import MDS 

mds=MDS(n_components=2, max_iter=100, metric=True, n_jobs=-1)

plotting(mds.fit_transform(train),train_label,'MDS')
from sklearn.decomposition import FastICA
fast_ica=FastICA(n_components=2, max_iter=50, algorithm='parallel')

plotting(fast_ica.fit_transform(train),train_label,'ICA')