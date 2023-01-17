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
dataset = pd.read_csv('../input/Mall_Customers.csv')

#getting information of dataset

dataset.head(10)
# Getting last 10 dataset

dataset.tail(10)
# information about total count and averages

dataset.describe()
# different properties of dataset

dataset.info()
# information about gender

dataset.Gender.value_counts()
dataset.hist(figsize = (18,12))

plt.show()
# Algorithm for clustring 

# Will be using K-Means to get better understanding

# x is independent variable having columns Age and Annual Income

x = dataset.iloc[:,[3,4]].values 



''' We will be required a centroids i.e this will allow us to group our data to perticular centroid for that we will use 

Elbow method of K-Means'''

# WCSS => within-cluster-sum-of-square

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)

y_means = kmeans.fit_predict(x)



# visualising the cluster

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'red', label = '1')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'green', label = '2')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'yellow', label = '3')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'blue', label = '4')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = '5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1] , s = 300, c = 'black', label = 'Centroid')

plt.title("Dataset analysis K-Means clustering")

plt.xlabel('Annual Income($k)')

plt.ylabel('Spending Scores')

plt.legend()

plt.show()