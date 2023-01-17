# -*- coding: utf-8 -*-

"""

Created on Fri Mar  6 22:30:11 2020



@author: ZIYA

"""





# 1.0 Call libraries

import pandas as pd

import numpy as np

import os

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

import seaborn as sns

from sklearn.metrics import silhouette_score

from warnings import filterwarnings

from numpy.linalg import norm

filterwarnings('ignore')

%matplotlib inline

    # 2.0 Set your working folder to where data is

os.chdir("../input/customer-segmentation-tutorial-in-python/")

os.listdir()
df = pd.read_csv("Mall_Customers.csv")
# 2.2 Explore dataset

pd.set_option("display.max_columns",50)

df.head()
df.shape               

df.dtypes

df.isnull().sum().sum()

# 2.3 Cleaning dataset

df.pop("CustomerID")

df.Gender[df.Gender=='Male']=1

df.Gender[df.Gender=='Female']=0

df.Gender=df.Gender.astype('int8')

df.rename({'Annual Income (k$)' : 'income','Spending Score (1-100)':'score'},

           axis = 1,

           inplace = True

         )
# 3.0 Plots: Pairplot

sns.pairplot(df,kind="reg")



#Conclusion

    #Age is giving Negative regression on Spending score.

    #Annual Income, Age & Spending score have some rough relation.

# 3.1 Plots: Distibution

sns.distplot(df.Age)

sns.distplot(df.income)



#Conclusion

    #Age: Major contribution from 30 to 40 age group & plot is skewed to left

    #Income: Major contribution is 50 to 80k earner.
# 3.2 Plots: Jointplot

sns.jointplot(df.Age, df.income, kind = 'reg')

sns.jointplot(df.Age, df.score, kind = 'reg')

sns.jointplot(df.income, df.score, kind = 'reg')



#Conclusion

    #As per the relation between Age,Income & Score they are carring more than 2 cluster.

    #Income & Score showing strong predictor than Age to Income.

#KMeans Clustering between Income & Score.



#Defualt value of K is 2 but here is code to find value of K manually to analysize the data better.



#df.pop('Gender')

x=df.iloc[:,[1,2]]

X_std = StandardScaler().fit_transform(x)

sse = []

list_k = list(range(1, 10))



for k in list_k:

    km = KMeans(n_clusters=k)

    km.fit(X_std)

    sse.append(km.inertia_)



# Plot sse against k

plt.figure(figsize=(6, 6))

plt.plot(list_k, sse, '-o')

plt.xlabel(r'Number of clusters *k*')

plt.ylabel('Sum of squared distance');



#Conclusion: We can take k=5 here.
#Defining KMeans Class (copy from google, dont have much knowledge)

class Kmeans:

    '''Implementing Kmeans algorithm.'''



    def __init__(self, n_clusters, max_iter=100, random_state=123):

        self.n_clusters = n_clusters

        self.max_iter = max_iter

        self.random_state = random_state



    def initializ_centroids(self, X):

        np.random.RandomState(self.random_state)

        random_idx = np.random.permutation(X.shape[0])

        centroids = X[random_idx[:self.n_clusters]]

        return centroids



    def compute_centroids(self, X, labels):

        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):

            centroids[k, :] = np.mean(X[labels == k, :], axis=0)

        return centroids



    def compute_distance(self, X, centroids):

        distance = np.zeros((X.shape[0], self.n_clusters))

        for k in range(self.n_clusters):

            row_norm = norm(X - centroids[k, :], axis=1)

            distance[:, k] = np.square(row_norm)

        return distance



    def find_closest_cluster(self, distance):

        return np.argmin(distance, axis=1)



    def compute_sse(self, X, labels, centroids):

        distance = np.zeros(X.shape[0])

        for k in range(self.n_clusters):

            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)

        return np.sum(np.square(distance))

    

    def fit(self, X):

        self.centroids = self.initializ_centroids(X)

        for i in range(self.max_iter):

            old_centroids = self.centroids

            distance = self.compute_distance(X, old_centroids)

            self.labels = self.find_closest_cluster(distance)

            self.centroids = self.compute_centroids(X, self.labels)

            if np.all(old_centroids == self.centroids):

                break

        self.error = self.compute_sse(X, self.labels, self.centroids)

    

    def predict(self, X):

        distance = self.compute_distance(X, old_centroids)

        return self.find_closest_cluster(distance)
km = Kmeans(n_clusters=5, max_iter=200)

km.fit(X_std)

centroids = km.centroids



#Plot the clustered data

fig, ax = plt.subplots(figsize=(6, 6))

plt.scatter(X_std[km.labels == 0, 0], X_std[km.labels == 0, 1],

            c='green', label='cluster 1')

plt.scatter(X_std[km.labels == 1, 0], X_std[km.labels == 1, 1],

            c='blue', label='cluster 2')

plt.scatter(X_std[km.labels == 2, 0], X_std[km.labels == 2, 1],

            c='pink', label='cluster 3')

plt.scatter(X_std[km.labels == 3, 0], X_std[km.labels == 3, 1],

            c='black', label='cluster 4')

plt.scatter(X_std[km.labels == 4, 0], X_std[km.labels == 4, 1],

            c='orange', label='cluster 5')





plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,

            c='r', label='centroid')

plt.legend()

plt.xlim([-2, 2])

plt.ylim([-2, 2])

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('Visualization for k', fontweight='bold')

ax.set_aspect('equal');