# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn.cluster import KMeans

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import DBSCAN 

from sklearn.preprocessing import StandardScaler 

from sklearn.cluster import Birch

from sklearn.cluster import MeanShift

from sklearn.mixture import GaussianMixture

from sklearn.cluster import AffinityPropagation

from sklearn.cluster import OPTICS

from sklearn.cluster import SpectralClustering

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans

data = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

X = data.iloc[:200, [2, 4]].values

def dbscan(X, eps, min_samples):

    ss = StandardScaler()

    X = ss.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)

    db.fit(X)

    y_pred = db.fit_predict(X)

    plt.figure(figsize=(10,10))

    plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='coolwarm')

dbscan(X,eps=0.275,min_samples=5)



ss = StandardScaler()

X = ss.fit_transform(X)

db = DBSCAN(eps=0.275, min_samples=5)

db.fit(X)

y_pred = db.fit_predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='coolwarm')



ag=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

ag.fit_predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=brc_y_pred, cmap='coolwarm')
brc = Birch(n_clusters=5)

brc.fit(X)

brc_y_pred = brc.predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=brc_y_pred, cmap='coolwarm')

gmm = GaussianMixture(n_components=5)

gmm.fit(X)

gmm_y_pred = gmm.predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=gmm_y_pred, cmap='coolwarm')



sc = SpectralClustering(n_clusters=5)

sc_y_pred = sc.fit_predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=sc_y_pred, cmap='coolwarm')

kmeans=KMeans(n_clusters=5)

kmeans.fit_predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=ap_y_pred, cmap='spring')
ap = AffinityPropagation(random_state=0)

ap.fit(X)

ap_y_pred = ap.predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=ap_y_pred, cmap='spring')

from sklearn.cluster import MiniBatchKMeans

mks = MiniBatchKMeans(n_clusters=2)

mks.fit(X)

mks_y_pred = mks.predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=mks_y_pred, cmap='seismic')

ms = MeanShift(bandwidth=2)

ms.fit(X)

ms_y_pred = ms.predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=ms_y_pred, cmap='seismic')



opt = OPTICS(min_samples=5)

opt_y_pred = opt.fit_predict(X)

plt.figure(figsize=(10,10))

plt.scatter(X[:,0], X[:,1],c=opt_y_pred, cmap='inferno')


