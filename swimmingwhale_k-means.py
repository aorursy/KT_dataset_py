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
ex7data2 = pd.read_csv("../input/ex7data2.csv",header=None)
ex7data2.head()
ex7data2.plot.scatter(x=0,y=1)
def findClosestCentroids(X, centroids):
    distance = np.zeros((len(X),len(centroids)))
    for i in range(len(X)):
        for j in range(len(centroids)):
            distance[i,j] = np.linalg.norm(X[i,:]-centroids[j,:])
    
    return np.argmin(distance,axis=1)
def computeCentroids(X, idx, K):
    centroids = np.zeros((K,X.shape[1]))
    for i in range(K):
        centroids[i,:] = np.mean(X[idx == i],axis = 0)
        
    return centroids
def runkMeans(X,K,max_iters):
    indexs = np.random.choice(np.array(range(len(X))), K,replace=False)
    centroids = X[indexs]
    for max_iter in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)
        colors = ['','','']
        for i in range(K):
            plt.scatter(X[idx==i, 0], X[idx==i, 1])
        plt.scatter(centroids[:, 0], centroids[:, 1], c='r')
        plt.show()
K = 3
max_iters = 3
runkMeans(ex7data2.values,K,max_iters)
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(ex7data2)
plt.scatter(ex7data2.iloc[:, 0], ex7data2.iloc[:, 1], c=y_pred)
plt.show()