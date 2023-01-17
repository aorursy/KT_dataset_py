# -*- coding: utf-8 -*-

"""

Created on Mon Dec 18 07:08:22 2017



@author: Sheetal

"""



#importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#importing the mall dataset

dataset=pd.read_csv("degrees.csv")

X=dataset.iloc[:,[2,3]].values



#using elbow method to find optimal number of clusters

from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):

    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)    

plt.plot(range(1,11),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('WCSS')

plt.show



#Applying K-Means to mall dataset

kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)

y_kmeans=kmeans.fit_predict(X)



#Visualizing our results

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Good Choice')

plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Okay Choice')

plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Poor Choice')

plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Best Choice')

plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Medium Choice')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=50,c='Yellow',label='Centroids')

plt.title('Clusters of Career Choice')

plt.xlabel('Starting Salary')

plt.ylabel('Mid Career salary in 90th percentile')

plt.legend()

plt.show()