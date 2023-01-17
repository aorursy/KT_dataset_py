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

%matplotlib inline

from sklearn.datasets import make_blobs
centers = [[1, 1], [-1, -1], [1, -1]]

X, y = make_blobs(n_samples=100, centers=centers,random_state=2,cluster_std=0.2)

fig,ax = plt.subplots(figsize=(10,10))

plt.xlabel('X1',fontsize=20)

plt.ylabel('X2',fontsize=20)

ax.set_title("Actual Data(3 clusters)",fontsize=30)

ax.scatter(X[:,0],X[:,1],c=y,s=50,cmap = 'spring');
eps = 1

minpts = 3

D = X



def update_labels(X,pt,eps,labels,cluster_val):

    neighbors = []

    label_index = []

    for i in range(X.shape[0]):

        if np.linalg.norm(X[pt]-X[i])<eps:

            neighbors.append(X[i])

            label_index.append(i)

    if len(neighbors) <minpts:

        for i in range(len(labels)):

            if i in label_index:

                labels[i]=-1

    else:

        for i in range(len(labels)):

            if i in label_index:

                labels[i]=cluster_val

    return labels



labels = [0]*X.shape[0]

C = 1

for p in range(X.shape[0]):

    if labels[p]==0:

        labels = update_labels(X,p,eps,labels,C)

        C= C+1

        

fig,ax = plt.subplots(figsize=(10,10))

plt.xlabel('X1',fontsize=20)

plt.ylabel('X2',fontsize=20)

ax.set_title("DBSCAN clustered data (self-implementation)",fontsize=30)

ax.scatter(X[:,0],X[:,1],c=labels,s=50,cmap = 'winter');
from sklearn.cluster import DBSCAN



dbscan = DBSCAN(eps = 0.6).fit(X)

dbscanlabels = dbscan.labels_

fig,ax = plt.subplots(figsize=(10,10))

plt.xlabel('X1',fontsize=20)

plt.ylabel('X2',fontsize=20)

ax.set_title("DBSCAN clustered data (Using sklearn library)",fontsize=30)

ax.scatter(X[:,0],X[:,1],c=dbscanlabels,s=50,cmap = 'Wistia');
from sklearn import metrics

print("Silhouette score: {:.2f}%".format(metrics.silhouette_score(X,labels)*100))