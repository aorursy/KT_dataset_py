# data loading and processing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as sci #using the pearson



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# model training and machine learning

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score # evaluate the model



# file operation

import os

print(os.listdir("../input"))



#ignore warnings

import warnings

warnings.filterwarnings("ignore")
# read the data and store data in DataFrame titled Mall_data

Mall_data = pd.read_csv(r"../input/Mall_Customers.csv")

# print a summary of the data in Melbourne data

Mall_data.info()
# Peak on the data

Mall_data.head()
#plotting with countplot

plt.subplot(1,2,1)

sns.countplot(x='Gender', data=Mall_data)

plt.title('Customer Distribution Of Gender')



#plotting with pie

plt.subplot(1,2,2)

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(x=Mall_data['Gender'].value_counts(),

        explode = explode, autopct = '%.2f%%')

plt.title('Gender Ratio', fontsize = 20)

plt.legend(labels = ['Female', 'Male'], loc='best')

plt.show()
Gender_Score = Mall_data.groupby('Gender')['Spending Score (1-100)'].agg([np.mean, max, min])

Gender_Score
#Creating the figure

figure = plt.figure(figsize=(15,6), dpi=150)



#Single Variable

Features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']



n=0

for i in Features:

    n += 1

    plt.subplot(1, 3, n)

    plt.subplots_adjust(wspace=1, right=1)

    sns.distplot(Mall_data[i], rug=True) #rug setting the observation strip;find out data's concentration

    plt.title("Distribution Of {}".format(i))
sns.swarmplot(data=Mall_data, x='Spending Score (1-100)', y='Gender')
sns.boxenplot(data=Mall_data, x='Gender', y='Spending Score (1-100)')
sns.jointplot(x='Age', y='Annual Income (k$)', data=Mall_data, kind='hex', stat_func=sci.pearsonr, ratio=5)
sns.jointplot(x='Annual Income (k$)', y='Spending Score (1-100)', 

              data=Mall_data, stat_func=sci.pearsonr, kind='scatter')
sns.pairplot(data=Mall_data, vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], \

             hue='Gender', kind='reg', diag_kind='kde', markers=['*','.'], size=5, palette='husl')
X1_Matrix = Mall_data.iloc[:, [2,4]].values # Age & Spending Score

X2_Matrix = Mall_data.iloc[:, [3,4]].values # Annual Income & Spending Score
inertias_1 = []

for i in range(1,8):

    kmeans = KMeans(n_clusters=i, init='k-means++',  max_iter=300, n_init=10,

                   random_state=0)

    kmeans.fit(X1_Matrix)

    inertia = kmeans.inertia_

    inertias_1.append(inertia)

    print('For n_cluster =', i, 'The inertia is:', inertia)
# Creating the figure

figure = plt.figure(1, figsize=(15,6), dpi=80)



plt.plot(np.arange(1,8), inertias_1, alpha=0.8, marker='o')
Kmeans = KMeans(n_clusters=5, init='k-means++',  max_iter=300, n_init=10,

                   random_state=0)

labels = Kmeans.fit_predict(X1_Matrix)
centroids1 = Kmeans.cluster_centers_ # the centroid points in each cluster
# Visualizing the 5 clusters

plt.scatter(x=X1_Matrix[labels==0, 0], y=X1_Matrix[labels==0, 1], s=120, c='red', label='Potential and should be stumulated')

plt.scatter(x=X1_Matrix[labels==1, 0], y=X1_Matrix[labels==1, 1], s=120, c='blue', label='Premium and should be retained')

plt.scatter(x=X1_Matrix[labels==2, 0], y=X1_Matrix[labels==2, 1], s=120, c='grey', label='Potential and should be treated carefully')

plt.scatter(x=X1_Matrix[labels==3, 0], y=X1_Matrix[labels==3, 1], s=120, c='orange', label='better than normal and should be paid more attention')

plt.scatter(x=X1_Matrix[labels==4, 0], y=X1_Matrix[labels==4, 1], s=120, c='green', label='normal and should be ovserved for a while')



#Visualizing every centroids in different cluster.

plt.scatter(x=centroids1[:,0], y=centroids1[:,1], s=300, alpha=0.8, c='yellow', label='Centroids')



#Style Setting

plt.title("Cluster Of Customers", fontsize=20)

plt.xlabel("Age")

plt.ylabel("Spending Score (1-100)")

plt.legend(loc=0)
pd.Series(labels).value_counts()
inertias_2 = []

for i in range(1,8):

    kmeans = KMeans(n_clusters=i, init='k-means++',  max_iter=300, n_init=10,

                   random_state=1)

    kmeans.fit(X2_Matrix)

    inertia = kmeans.inertia_

    inertias_2.append(inertia)

    print('For n_cluster =', i, 'The inertia is:', inertia)
# Creating the figure

figure = plt.figure(1, figsize=(15,6), dpi=80)



plt.plot(np.arange(1,8), inertias_2, alpha=0.8, marker='o')
Kmeans = KMeans(n_clusters=5, init='k-means++',  max_iter=300, n_init=10,

                   random_state=1)

labels = Kmeans.fit_predict(X2_Matrix)
centroids2 = Kmeans.cluster_centers_ # the centroid points in each cluster
# Visualizing the 5 clusters

plt.scatter(x=X2_Matrix[labels==0, 0], y=X2_Matrix[labels==0, 1], s=120, c='red', label='normal income and hedonism')

plt.scatter(x=X2_Matrix[labels==1, 0], y=X2_Matrix[labels==1, 1], s=120, c='blue', label='high income and frugalism')

plt.scatter(x=X2_Matrix[labels==2, 0], y=X2_Matrix[labels==2, 1], s=120, c='grey', label='medium income and pragmatism')

plt.scatter(x=X2_Matrix[labels==3, 0], y=X2_Matrix[labels==3, 1], s=120, c='orange', label='high income and hedonism')

plt.scatter(x=X2_Matrix[labels==4, 0], y=X2_Matrix[labels==4, 1], s=120, c='green', label='normal income and frugalism')



#Visualizing every centroids in different cluster.

plt.scatter(x=centroids2[:,0], y=centroids2[:,1], s=300, alpha=0.8, c='yellow', label='Centroids')



#Style Setting

plt.title("Cluster Of Customers", fontsize=20)

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.legend(loc=7)
for n_clusters in range(3,8):

    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)

    kmeans.fit(X2_Matrix)

    clusters = kmeans.predict(X2_Matrix)

    silhouette_avg = silhouette_score(X2_Matrix, clusters)

    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
n_clusters = 5

kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)

kmeans.fit(X2_Matrix)

clusters = kmeans.predict(X2_Matrix)

silhouette_avg = silhouette_score(X2_Matrix, clusters)

print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
pd.Series(clusters).value_counts()