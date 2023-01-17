import numpy as np
import scipy as sp
import scipy.linalg as linalg
import sklearn.neighbors as nb
import sklearn.utils.graph as ug
from time import time
from numpy import *
from sklearn.datasets import load_iris
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn.preprocessing import LabelEncoder

from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from scipy.spatial import distance as spd
%matplotlib inline
#data : the data matrix
#k the number of component to return
#return the new data and the variance that was maintained 
def pca1(data,k):
	# Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
	# Rows of A correspond to observations (wines), columns to variables.
	## TODO: Implement PCA

	# compute the mean

    m = np.mean(data,0)
	# subtract the mean (along columns)
    C = data-m
	# compute covariance matrix
    cov_mat = np.dot(transpose(C),C)
	# compute eigenvalues and eigenvectors of covariance matrix
    eigval,eigvect = linalg.eig(cov_mat)
	# Sort eigenvalues (their indexes)
    idx = eigval.argsort()[::-1]
    eigval[idx]
	# Sort eigenvectors according to eigenvalues
    vect = eigvect[:,idx]
    eigk = vect[:,0:k]
	# Project the data to the new space (k-D) and measure how much variance we kept
    data2 = np.dot(C,eigk)
    perc = sum(eigval[0:k])/sum(eigval)
    return  (data2, perc)#put here a tuple to return both 
#D :distance matrix
#k: number of vectors to use
def mds(D,k):
    nelem = D.shape[0]
    J = eye(nelem) - (1.0/nelem) * ones(nelem)
    # Compute matrix B
    B = -(1.0/2) * dot(J,dot(pow(D,2),J))
    # SVD decomposition of B  
    U,L,V = linalg.svd(B)
    return dot(U[:,:k],sqrt(diag(L)[:k,:k]))

def isomap(D,k,n_neighbors):
    #k nearest neighbour algorithm
    knr = nb.NearestNeighbors(n_neighbors= n_neighbors)
    knr.fit(D)
    #neighbour graph where the edges are weighted with the euclidean distance
    kng = nb.kneighbors_graph(knr,n_neighbors,mode='distance')
    #graph distances 
    Dist_g = ug.graph_shortest_path(kng,directed=False,method='auto')
    #the rest is just like mds or PCA (if we want)
    nelem = D.shape[0]
    J = np.eye(nelem) - (1.0/nelem) * np.ones(nelem)
    # Compute matrix B
    B = -1.0/2 * np.dot(J,np.dot(pow(Dist_g,2),J))
    # SVD decomposition of B  
    U,L,V = linalg.svd(B)
    return np.dot(U[:,:k],np.sqrt(diag(L)[:k,:k]))
def LDA(X, Y):
    classLabels = np.unique(Y)
    classNum = len(classLabels)
    datanum, dim = X.shape
    totalMean = np.mean(X,0)

	# partition class labels per label - list of arrays per label

    partition = [np.where(Y==label) for label in classLabels]

	# find mean value per class (per attribute) - list of arrays per label
    classMean = [(np.mean(X[idx],0),len(idx)) for idx in partition]

	# Compute the within-class scatter matrix
    Sw = np.zeros((dim,dim))
	# covariance matrix of each class * fraction of instances for that class 
    for idx in partition:
        Sw=Sw + np.cov(X[idx],rowvar = 0) * len(idx)
	# Compute the between-class scatter matrix
    Sb = np.zeros((dim,dim))
    for class_mean,class_size in classMean:
        temp=(class_mean-totalMean)[:,np.newaxis]
        Sb=Sb+class_size*np.dot(temp,np.transpose(temp))


	# Solve the eigenvalue problem for discriminant directions to maximize class separability while simultaneously minimizing
	# the variance within each class
    T=np.dot(linalg.inv(Sw),Sb)
    eigval, eigvec = linalg.eig(T) 


    idx = eigval.argsort()[::-1] # Sort eigenvalues
    eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
    W = np.real(eigvec[:,:classNum-1]) # eigenvectors correspond to k-1 largest eigenvalues


	# Project data onto the new LDA space
    X_lda = np.dot(X,W)

	# project the mean vectors of each class onto the LDA space
    projected_centroid = [np.dot(m,W) for m,class_size in classMean]

    return W, projected_centroid, X_lda
# The next line is required for plotting only
Axes3D

iris = pd.read_csv('../input/Iris.csv')
X = np.array(iris[[c for c in iris.columns if c != "Species" and c!='Id']])
Y_iris = iris["Species"]
color = LabelEncoder().fit_transform(Y_iris)


n_components=2
n_neighbors=5

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

#------PCA--------our implementation
t0 = time()
(Y,perc)=pca1(X,n_components)
t1 = time()
print("PCA(imp): %.2g sec" % (t1 - t0))
ax = fig.add_subplot(252)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("PCA(imp) (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#-----------------

#------MDS--------our implementation (classical MDS)
t0 = time()
D=spd.squareform(spd.pdist(X,'euclidean'))
Y=mds(D,n_components)
t1 = time()
print("MDS(imp): %.2g sec" % (t1 - t0))
ax = fig.add_subplot(253)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("MDS(imp) (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#-----------------
#------isomap--------our implementation (with MDS)
t0 = time()
Y=isomap(X,n_components,n_neighbors)
t1 = time()
print("Isomap(imp): %.2g sec" % (t1 - t0))
ax = fig.add_subplot(254)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Isomap(imp) (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#-----------------

#----PCA---------- sklearn implementation 

t0 = time()
pca = PCA(n_components=n_components)
Y = pca.fit_transform(X)
t1 = time()
print("PCA: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("PCA (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#--------------------
#----MDS---------- sklearn implementation (Stress minimization-majorization algorithm SMACOF)
t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#---------------
#----Isomap---------- sklearn implementation (with kernel PCA)
t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#--------------------



plt.show()
c=['r','g','b','k']
colors=[c[int(i-1)] for i in color]
# get LDA projection
W, projected_centroids, X_lda = LDA(X, color)

#perform PCA to compare with LDA
pca = PCA(n_components=2)
Y = pca.fit_transform(X)

#PLOT them side by side
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(121)
ax.scatter(X_lda[:,0], X_lda[:,1],color=colors)
for ar in projected_centroids:
	ax.scatter(ar[0], ar[1],color='k',s=100)
ax = fig.add_subplot(122)
ax.scatter(Y[:,0], Y[:,1],color=colors)
plt.show()

