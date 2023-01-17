# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings  

warnings.filterwarnings("ignore")   # ignore warnings





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Mall_Customers.csv')
data.info()
data.head()
X = data.iloc[:, 3:].values

X
# Loading Library

from sklearn.cluster import KMeans
# To decide variable K, we use WCSS.

result = []

for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init='k-means++', random_state = 123)

    kmeans.fit(X)

    result.append(kmeans.inertia_)
result
plt.plot(range(1,11), result)

plt.xlabel('the number of clusters')

plt.ylabel('result')

plt.title('The Elbow Method')

plt.show()

# From below figure, we can choose K as 3 because there is a decreasing of acceleration at that point. 
# Applying K-Means Algorithm

# Creation of model

kmeans = KMeans(n_clusters = 3, init='k-means++')

kmeans.fit(X)
# We learn center point of each cluster with below function.

# For example, first column is Annual Income (k$) whose center point is 44.15447154 for first cluster.

kmeans.cluster_centers_
# Prediction

y_kmeans = kmeans.fit_predict(X)

y_kmeans
# Visualising the clusters

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=100, c='red')

plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=100, c='blue')

plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s=100, c='green')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow')

plt.title('K-Means Clustering')

plt.show()
# Creation of model & prediction

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

Y_predict = ac.fit_predict(X)

Y_predict
# Visualising the clusters

plt.scatter(X[Y_predict==0,0], X[Y_predict==0,1], s=100, c='red')

plt.scatter(X[Y_predict==1,0], X[Y_predict==1,1], s=100, c='blue')

plt.scatter(X[Y_predict==2,0], X[Y_predict==2,1], s=100, c='green')

plt.title('Hierarchical Clustering')

plt.show()
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.show()