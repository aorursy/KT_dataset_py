# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans



mallCustomerPATH="../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv"

mallCustomerDATA=pd.read_csv(mallCustomerPATH)

#mallCustomerDATA.describe()

#mallCustomerDATA.shape

#mallCustomerDATA.dtypes

#mallCustomerDATA.isnull().sum()

#sns.distplot(mallCustomerDATA['Age'])

#sns.countplot(mallCustomerDATA['Gender'])



algorithm = (KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300) )

X = mallCustomerDATA.iloc[:, [3,4]].values



algorithm.fit(X)

algorithm.labels_



   

kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)

y_kmeans= kmeansmodel.fit_predict(X)



plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'pink', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'gray', label = 'Cluster 3')

plt.scatter(algorithm.cluster_centers_[:, 0], algorithm.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')



plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()