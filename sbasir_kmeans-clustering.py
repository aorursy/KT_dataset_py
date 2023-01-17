# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



dataset = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

print(len(dataset))

dataset.head()
dataset.info()
dataset.isnull().sum()
plt.figure(1 , figsize = (17 , 8))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    sns.distplot(dataset[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
plt.figure(1 , figsize = (10 , 5))

sns.countplot(y = 'Gender' , data = dataset)

plt.show()
### Feature sleection for the model

#Considering only 2 features (Annual income and Spending Score) and no Label available

x = dataset.iloc[:, [3,4]].values

#print(x)
#Finding the possible clusters

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

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

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1:Red')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Cluster 2:Yellow')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3:Green')

plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'violet', label = 'Cluster 4:Violet')

plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'black', label = 'Cluster 5:Cyan')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'navy', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()