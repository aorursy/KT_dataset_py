# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/Mall_Customers.csv')
dataset.head()
dataset.rename(columns={'Annual Income (k$)':'Annual_Income','Spending Score (1-100)':'Spending_Score'},inplace=True)
dataset.shape
dataset.head()
dataset.describe()
dataset.info()
dataset.nunique()
plt.figure(figsize=(15,5))

sns.countplot(dataset['Age'])

plt.title('Age Distribution')

plt.xlabel('Age')

plt.show()
plt.figure(figsize=(20,10))

sns.countplot(dataset['Annual_Income'])

plt.title('Annual Income')

plt.xlabel('Annual Income($)')

plt.show()
plt.figure(figsize=(20,8))

sns.countplot(dataset['Spending_Score'])

plt.title('Spending Score Distribution')

plt.xlabel('Spending Score')

plt.ylabel('Count')

plt.axis()

plt.show()
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(dataset['Age'])

plt.title('Age Distribution')

plt.xlabel('Age')

plt.ylabel('Count')



plt.subplot(1,2,2)

sns.distplot(dataset['Annual_Income'],color='pink')

plt.title('Annual Income Distribution')

plt.xlabel('Annual Income')

plt.ylabel('Count')
plt.figure(figsize=(8,8))



colors = ['LightBlue','Lightgreen']

explode = [0,0.1]

plt.pie(dataset['Gender'].value_counts(),explode=explode,colors=colors,autopct='%1.1f%%',shadow=True,startangle=140)

plt.legend(labels=['Female','Male'])

plt.title('Male v/s Female Distribution')

plt.axis('off')
dataset['Gender'].value_counts()
plt.figure(figsize=(15,8))

sns.heatmap(dataset.corr(),annot=True)

plt.show()
plt.figure(figsize=(15,5))

sns.stripplot(dataset['Gender'], dataset['Spending_Score'])

plt.title('Strip plot for Gender vs Spending Score')

plt.show()



plt.figure(figsize=(15,5))

sns.boxplot(dataset['Gender'], dataset['Spending_Score'])

plt.title('Box plot for Gender vs Spending Score')

plt.show()



plt.figure(figsize=(15,5))

sns.violinplot(dataset['Gender'],dataset['Spending_Score'])

plt.title('Gender Wise Spending Score')

plt.show()



plt.figure(figsize=(15,5))

sns.violinplot(dataset['Gender'],dataset['Annual_Income'])

plt.title('Gender wise Annual Income Distribution')

plt.show()



plt.figure(figsize=(15,5))

sns.boxplot(dataset['Gender'],dataset['Annual_Income'])

plt.title('Gender wise Annual Income Distribution')

plt.show()



x = dataset.loc[:,['Age' , 'Annual_Income']].values
from sklearn.cluster import KMeans



wcss =[]

for n in range(1,11):

    kmeans=KMeans(n_clusters=n,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)



plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.grid()

plt.show()
#Applying K Means 



kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0,algorithm='elkan')

y_means = kmeans.fit_predict(x)
#Visualizing

plt.figure(figsize=(15,10))

plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='red',label='cluster1')

plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='yellow',label='cluster2')

plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='green',label='cluster3')

plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c='blue',label='cluster4')



plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='centroids')

plt.title('Age wise Annual Income Cluster')

plt.xlabel('Age')

plt.ylabel('Annual Income($)')

plt.legend()

plt.show()
x1 = dataset.loc[:,['Age','Spending_Score']].values
from sklearn.cluster import KMeans



wcss =[]

for n in range(1,11):

    kmeans=KMeans(n_clusters=n,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(x1)

    wcss.append(kmeans.inertia_)



plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
#Applying K Means 



kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)

y_means = kmeans.fit_predict(x1)
#Visualizing

plt.figure(figsize=(10,10))

plt.scatter(x1[y_means==0,0],x1[y_means==0,1],s=100,c='red',label='cluster1')

plt.scatter(x1[y_means==1,0],x1[y_means==1,1],s=100,c='yellow',label='cluster2')

plt.scatter(x1[y_means==2,0],x1[y_means==2,1],s=100,c='green',label='cluster3')

plt.scatter(x1[y_means==3,0],x1[y_means==3,1],s=100,c='blue',label='cluster4')



plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='centroids')

plt.title('Age wise Spending Score(0-100)')

plt.xlabel('Age')

plt.ylabel('Spending Score(0-100)')

plt.legend()
#Feature Selection for Annual Income and Spending Score

x2 = dataset.loc[:,['Annual_Income','Spending_Score']].values
from sklearn.cluster import KMeans



wcss =[]

for n in range(1,11):

    kmeans=KMeans(n_clusters=n,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(x2)

    wcss.append(kmeans.inertia_)



plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of Clusters')

plt.ylabel('wcss')

plt.show()
# If we zoom the plot above , optimal clusters comes to 5, so we will take it n_clusters as 5 here.

kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)

y_kmeans= kmeansmodel.fit_predict(x2)
#Visualizing all the clusters 



plt.figure(figsize=(15,8))

plt.scatter(x2[y_kmeans == 0, 0], x2[y_kmeans == 0, 1], s = 100, c = 'Orange', label = 'Pinch Penny')

plt.scatter(x2[y_kmeans == 1, 0], x2[y_kmeans == 1, 1], s = 100, c = 'Pink', label = 'Normal Customer')

plt.scatter(x2[y_kmeans == 2, 0], x2[y_kmeans == 2, 1], s = 100, c = 'yellow', label = 'Target Customer')

plt.scatter(x2[y_kmeans == 3, 0], x2[y_kmeans == 3, 1], s = 100, c = 'magenta', label = 'Spender')

plt.scatter(x2[y_kmeans == 4, 0], x2[y_kmeans == 4, 1], s = 100, c = 'Red', label = 'Balanced Customer')



plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'Black', label = 'Centroids')

plt.title('Annual Income v/s Spending Score')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
