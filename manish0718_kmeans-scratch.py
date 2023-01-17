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
# Generate data
X, y = make_blobs(n_samples=300, centers=5)
print (X.shape, y.shape)
plt.figure(0)
plt.grid(True)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
class KMeans:
    def __init__(self,k=5,max_iter=100):
        self.k=k
        self.max_iter=max_iter
    def fit(self,data,k=5,max_iter=100):
        self.means=[]
        #randomly initialize the means
        for i in range(self.k):
            self.means.append(data[i])
        for i in range(self.max_iter):
            #assign the data points that they belong to
            #create empty clusters
            clusters=[]
            for j in range(self.k):
                clusters.append([])
            for point in data:
                #find distance to all the means values
                distances=[((point-m)**2).sum() for m in self.means]
                #find the minimum distance
                minDistance=min(distances)
                #find the mean for which we got the min distance
                l=distances.index(minDistance)
                #add this point to the cluster
                clusters[l].append(point)
        
        
            #calculate the new mean valuesL
            change=False
            for j in range(self.k):
                new_mean=np.average(clusters[j],axis=0)
                if not np.array_equal(self.means[j],new_mean):
                    change=True
                self.means[j]=new_mean
            if not change:
                break

    
    def predict(self,test_data):
        predictions=[]
        for point in test_data:
           #find distance to all the means values
            distances=[((point-m)**2).sum() for m in self.means]
            #find the minimum distance
            minDistance=min(distances)
            #find the mean for which we got the min distance
            l=distances.index(minDistance)  
            predictions.append(l)
        return predictions

kmeans=KMeans()
kmeans.fit(X)
kmeans.predict(X)
kmeans.means
