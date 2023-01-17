# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import the libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Data Visualization 

import seaborn as sns  #Python library for Vidualization





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Import the dataset



dataset = pd.read_csv('../input/Mall_Customers.csv')



#Exploratory Data Analysis

#As this is unsupervised learning so Label (Output Column) is unknown



dataset.head(10) #Printing first 10 rows of the dataset

#total rows and colums in the dataset

dataset.shape
dataset.info() # there are no missing values as all the columns has 200 entries properly
#Missing values computation

dataset.isnull().sum()
### Feature selection for the model

#Considering only 2 features (Annual income and Spending Score) and no Label available

X= dataset.iloc[:, [3,4]].values

#Building the Model

#KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod

#to figure out K for KMeans, I will use ELBOW Method on KMEANS++ Calculation

from sklearn.cluster import KMeans

wcss=[]



#we always assume the max number of cluster would be 10

#you can judge the number of clusters by doing averaging

###Static code to get max no of clusters



for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)



    #inertia_ is the formula used to segregate the data points into clusters
#Visualizing the ELBOW method to get the optimal value of K 

plt.plot(range(1,11), wcss)

plt.title('The Elbow Method')

plt.xlabel('no of clusters')

plt.ylabel('wcss')

plt.show()
#If you zoom out this curve then you will see that last elbow comes at k=5

#no matter what range we select ex- (1,21) also i will see the same behaviour but if we chose higher range it is little difficult to visualize the ELBOW

#that is why we usually prefer range (1,11)

##Finally we got that k=5



#Model Build

kmeansmodel = KMeans(n_clusters= 5, init='k-means++',max_iter = 300, n_init = 10, random_state=0)



y_means= kmeansmodel.fit_predict(X)



#For unsupervised learning we use "fit_predict()" wherein for supervised learning we use "fit_tranform()"

#y_kmeans is the final model . Now how and where we will deploy this model in production is depends on what tool we are using.

#This use case is very common and it is used in BFS industry(credit card) and retail for customer segmenattion.
#Visualizing all the clusters 



plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

# I have changed centroids

centers = kmeansmodel.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=400, alpha=0.7, label = 'Centroids');

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()

###Model Interpretation 

#Cluster 1 (Red Color) -> earning high but spending less

#cluster 2 (Blue Colr) -> average in terms of earning and spending 

#cluster 3 (Green Color) -> earning high and also spending high [TARGET SET]

#cluster 4 (cyan Color) -> earning less but spending more

#Cluster 5 (magenta Color) -> Earning less , spending less





######We can put Cluster 3 into some alerting system where email can be send to them on daily basis as these re easy to converse ######

#wherein others we can set like once in a week or once in a month



# Thank you and please upvote for the motivation
x= dataset.iloc[:, [2,4]].values

wcss=[]

for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)

plt.title('The Elbow Method')

plt.xlabel('no of clusters')

plt.ylabel('wcss')

plt.show()
kmeansmodel = KMeans(n_clusters= 4, init='k-means++',max_iter = 300, n_init = 10, random_state=0)



y_means= kmeansmodel.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')



# I have changed centroids

centers = kmeansmodel.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=400, alpha=0.7, label = 'Centroids');

plt.title('Clusters of customers')

plt.xlabel(' Age ')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()