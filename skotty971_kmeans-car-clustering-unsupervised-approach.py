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
# K-Means Clustering



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/cars.csv')

dataset.head()
#Select only numeric and independ variable

X = dataset.iloc[:,:-1].values

#We have now an numpy ndarray we need to transform it into a dataframe

X=pd.DataFrame(X)

X = X.convert_objects(convert_numeric=True)

#Colonne Rename

X.columns = ['mpg', ' cylinders', ' cubicinches', ' hp', ' weightlbs', ' time-to-60', 'year']

X.describe()
#Eliminate null values

X.isnull().sum()
for i in X.columns:

    X[i] = X[i].fillna(int(X[i].mean()))

for i in X.columns:

    print(X[i].isnull().sum())

    


# Using the elbow method to find  the optimal number of clusters

from sklearn.cluster import KMeans



wcss=[]



for i in range (1,11):

    kmeans=KMeans(n_clusters=i,init='k-means++', max_iter=100, random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

# Applying k-means to the cars dataset

kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=100,random_state=0) 

y_kmeans = kmeans.fit_predict(X)



X = X.as_matrix(columns=None)



# Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],s=100,c='red',label='US')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],s=100,c='blue',label='Japan')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1],s=100,c='green',label='Europe')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')

plt.title('Clusters of car brands')

plt.legend()

plt.show()