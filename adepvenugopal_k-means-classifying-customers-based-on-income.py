##Importing the packages

#Data processing packages

import numpy as np 

import pandas as pd 



#Visualization packages

import matplotlib.pyplot as plt 

import seaborn as sns 



from sklearn.cluster import KMeans

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering



import warnings

warnings.filterwarnings('ignore')
#Import Mall Customer data

data = pd.read_csv('../input/Mall_Customers.csv')
#Find the size of the data Rows x Columns

data.shape
#Display first 5 rows of the data

data.head()
#Find Basic Statistics like count, mean, standard deviation, min, max etc.

data.describe()
#Find the the information about the fields, field datatypes and Null values

data.info()
plt.figure(figsize=(12,9))

plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], s = 25) #Point size is 25

plt.title('Raw Data',fontsize=15)

plt.xlabel('Annual Income (k$)',fontsize=15)

plt.ylabel('Spending Score (1-100)',fontsize=15)

plt.show()
#Extract Annual Income (k$) and Spending Score (1-100) fields 

target = data.iloc[:,[3,4]]
#Convert to Dataframe to  numpy array

X = np.array(target)
#Finding kmeans using no. of clusters = 5

kmeans = KMeans(n_clusters = 5, max_iter = 500, n_init = 10, random_state = 0)

kmeans_preds = kmeans.fit_predict(X)
point_size = 25

colors = ['red', 'blue', 'green', 'cyan', 'magenta']

labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']



plt.figure(figsize = (12,9))

for i in range(5):

    plt.scatter(X[kmeans_preds == i,0], X[kmeans_preds == i,1], s = point_size, c = colors[i], label = labels[i])

    

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 200, c = 'orange', label = 'Centroids')

plt.title('Clusters of Clients',fontsize=15)

plt.xlabel('Annual Income (k$)',fontsize=15)

plt.ylabel('Spending Score (1-100)',fontsize=15)

plt.legend(loc = 'best')

plt.show()