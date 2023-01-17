# Importing the libraries

import pandas as pd

from pandas import DataFrame

import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import warnings

pd.options.mode.chained_assignment = None

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=9,6

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
'''import os

print(os.getcwd())

os.chdir('D:\\DS_Notes\\Datasets_new\\')

# Importing the dataset

dataset= pd.read_csv('Mall_Customers.csv')

# displaying the data universities.csv

dataset.head()'''
dataset = pd.read_csv("../input/Mall_Customers.csv")

dataset.head()
X = dataset.iloc[:, [3, 4]].values

# printing first 6 rows of the dataset

X[:6]
plt.scatter(dataset['Annual Income (k$)'],dataset['Spending Score (1-100)'],color='blue',s=20)

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend(loc='best')

rcParams['figure.figsize']=8,6

plt.show()
# Fitting K-Means to the dataset 

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 40)

# Compute cluster centers and predict cluster index for each sample.

y_kmeans = kmeans.fit(X)

y_kmeans
y_kmeans = kmeans.fit_predict(X)

y_kmeans
kmeans.cluster_centers_
# Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label =

'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend(loc='best')

plt.show()
# naming each cluster based on the income and the amount they spend.

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'sensible')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'standard')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Target')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'careful')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'careless')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label ='Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
# add a new column and assign the cluster values to it

dataset["clusters"] = kmeans.labels_

dataset.head()
centers = pd.DataFrame(kmeans.cluster_centers_)

centers 
centers["clusters"] = [0,1,2,3,4]

#centers["clusters"] = range(5) 

centers
# performing merge operation as there are two similar columns named " clusters"

dataset = dataset.merge(centers)

dataset.head(10)
s1_grps = pd.DataFrame(kmeans.labels_)

s1_grps.cloumns = ('clusters')

#s1_grps.rename(columns={'0': 'cluster'}, inplace=True)

#s1_grps = s1_grps.rename(columns={'0':'Cluster'},inplace=False)

s1_grps.head()
s2_univs = dataset.iloc[:,0]

#s2_univs = dataset['CustomerID']

s2_univs.head()
rslt = pd.concat([s1_grps,s2_univs],axis=1)

rslt.rename(columns={'0':'Cluster'}, inplace=True)

#rslt['s2_univs'] = rslt.s2_univs.astype(int)

rslt.head(10)

#list(rslt)
dataset = pd.read_csv("../input/Mall_Customers.csv")

dataset.head()

# Now consider three input values

d = dataset.iloc[:, [2, 3, 4]]

d.head()
X = dataset.iloc[:, [2, 3, 4]].values

X
# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)

dataset["clusters"] = kmeans.labels_

dataset.head()
centers = pd.DataFrame(kmeans.cluster_centers_)

centers
centers = pd.DataFrame(kmeans.cluster_centers_)

centers["clusters"] = range(5) #n_clusters

dataset = dataset.merge(centers)

dataset