import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Graphs & Visualization 

import seaborn as sns



import os

print(os.listdir("../input"))

dataset = pd.read_csv('../input/Mall_Customers.csv')
#Let's check the data

dataset.head()
#Let's check the shape of data

dataset.shape
#Let's check datatypes

dataset.dtypes
dataset.isnull().sum()
plt.figure(1 , figsize = (17 , 8))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    sns.distplot(dataset[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
plt.figure(1 , figsize = (17 , 8))

sns.countplot(y = 'Gender' , data = dataset)

plt.show()
### Feature sleection for the model

#Considering only 2 features (Annual income and Spending Score) and no Label available

x = dataset.iloc[:, [3,4]].values
print(x)
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#KMeans is our Algorithms which provided in SKlearn

#n_clusters is a nummber of clusters which we will define 

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

#Let's predict the x

y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)

#We convert our prediction to dataframe so we can easily see this prediction in table form

df_pred = pd.DataFrame(y_kmeans)

df_pred.head()
plt.figure(1 , figsize = (17 , 8))

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Cluster 2')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'aqua', label = 'Cluster 3')

plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'violet', label = 'Cluster 4')

plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'lightgreen', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'navy', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
#Cluster 1 (Red Color) -> Earning medium but spending medium

#cluster 2 (Yellow Colr) -> Earning High but spending very less 

#cluster 3 (Aqua Color) -> Earning is low & spending is low

#cluster 4 (Violet Color) -> Earning is less but spending more -> Mall can target this type of people

#Cluster 5 (Lightgereen Color) -> Earning High & spending more -> Mall can target this type of people

#Navy color small circles is our Centroids
plt.figure(1 , figsize = (17 , 8))

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Standard people')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Tightwad people')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'aqua', label = 'Normal people')

plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'violet', label = 'Careless people(TARGET)')

plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'lightgreen', label = 'Rich people(TARGET)')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'navy', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()