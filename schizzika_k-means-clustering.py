# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

df.head()
df.isnull().any()
df.shape

      
df.describe()
df.dtypes
X = df.iloc[:, [3, 4]].values
type(X)
X
#Visualise data points

plt.scatter(X[:, 0], X[:,1], marker = '+')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score out of 10')

plt.show()
X[:,1] #Spending score
X[:, 0] #Annual income
#KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod

#to figure out K for KMeans, I will use ELBOW Method on KMEANS++ Calculation

from sklearn.cluster import KMeans

SSE = []



#we always assume the max number of cluster would be 10

#you can judge the number of clusters by doing averaging

###Static code to get max no of clusters



for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(X)

    SSE.append(kmeans.inertia_)



    #inertia_ is the formula used to segregate the data points into clusters
SSE
#Visualizing the ELBOW method to get the optimal value of K 



plt.plot(range(1,11), wcss)

plt.title('The Elbow Method')

plt.xlabel('No of clusters')

plt.ylabel('SSE')

plt.show()
#If you zoom out this curve then you will see that last elbow comes at k=5

#no matter what range we select ex- (1,21) also i will see the same behaviour but if we chose higher range it is little difficult to visualize the ELBOW

#that is why we usually prefer range (1,11)

##Finally we got that k=5



#Model Build

kmeansmodel = KMeans(n_clusters = 5, init ='k-means++', random_state = 0)

y_kmeans = kmeansmodel.fit_predict(X)



#For unsupervised learning we use "fit_predict()" wherein for supervised learning we use "fit_tranform()"

#y_kmeans is the final model . Now how and where we will deploy this model in production is depends on what tool we are using.

#This use case is very common and it is used in BFS industry(credit card) and retail for customer segmenattion.

type(y_kmeans)
y_kmeans
#Visualizing all the clusters 



plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
print('[0, 0] = ', X[y_kmeans == 0, 0])

print('[0, 1] = ', X[y_kmeans == 0, 1])

print('[1, 0] = ', X[y_kmeans == 1, 0])

print('[1, 1] = ', X[y_kmeans == 1, 1])

print('[2, 0] = ', X[y_kmeans == 2, 0])

print('[2, 1] = ', X[y_kmeans == 2, 1])

print('[3, 0] = ', X[y_kmeans == 3, 0])

print('[3, 1] = ', X[y_kmeans == 3, 1])

print('[4, 0] = ', X[y_kmeans == 4, 0])

print('[4, 1] = ', X[y_kmeans == 4, 1])

print(kmeans.cluster_centers_[:, 0])

print(kmeans.cluster_centers_[:, 1])