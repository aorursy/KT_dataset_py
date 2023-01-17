# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import make_blobs
#generate data
X,y=make_blobs(n_samples=200,centers=5)
print(X.shape,y.shape)
#plotting the data
plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1])
plt.show()
#import K-means from sklearn
from sklearn.cluster import KMeans
clf=KMeans(n_clusters=5)
clf.fit(X)
#cluster labels
clf.labels_
#mean points of the cluster
z=clf.cluster_centers_
print(z)
#plotting the clusters with their mean centers
plt.scatter(X[:,0],X[:,1],c=clf.labels_)
plt.scatter(z[:,0],z[:,1],c='blue')
plt.show()
