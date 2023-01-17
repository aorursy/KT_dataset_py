# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import defaultdict

from datetime import datetime

from scipy import stats

from statsmodels.formula.api import ols

import seaborn

import sklearn

from sklearn.decomposition import RandomizedPCA, PCA, SparsePCA

from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

from sklearn.preprocessing import StandardScaler

from scipy import linalg as LA

m = pd.read_csv('../input/roboBohr.csv')

m.drop('pubchem_id', axis = 1)

X = m.drop(['Eat', 'pubchem_id'],axis=1).values

X = StandardScaler().fit_transform(X)

Y = m['Eat']
pca = PCA()

Xt = pca.fit_transform(X)

pca_score = pca.explained_variance_ratio_

V = pca.components_
fig = plt.figure(figsize=(16, 6))

ax1 = fig.add_subplot(111)

ax1.set_ylim([0,1])

lin1 = ax1.scatter(range(0, int(pca_score.shape[0])), pca_score, c = 'b', label = 'no random noise')

plt.show()
fig = plt.figure(figsize=(16, 6))

ax2 = fig.add_subplot(111)

ax2.scatter(Xt[:,0], Xt[:, 1], c=Y)

plt.show()
#Singular value decomposition of the covariance matrix

cov = np.cov(X, rowvar = False)

evals , evecs = LA.eigh(cov)



#Sort the eigenvectors based on the eigenvalues

idx = np.argsort(evals)[::-1]

evecs = evecs[:,idx]

evals = evals[idx]



#Transform the data

Xt2 = np.dot(X, evecs)
fig = plt.figure(figsize=(16, 6))

ax2 = fig.add_subplot(111)

ax2.scatter(Xt2[:,0], Xt2[:, 1], c=Y)

plt.show()
# I tried dropping the first column of, but it didn't help much

X = X[:,1:]

pca = PCA()

Xt = pca.fit_transform(X)

fig = plt.figure(figsize=(16, 6))

ax2 = fig.add_subplot(111)

ax2.scatter(Xt[:,0], Xt[:, 1], c=Y)

plt.show()