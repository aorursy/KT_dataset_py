import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans
path='../input/Mall_Customers.csv'

dl=pd.read_csv(path)

dl.head()
dl.columns
dl.shape
dl.describe()
dl.nunique()
dl.dtypes
dl.isnull().sum()
plt.rcParams['figure.figsize']=(5,5)

hm=sns.heatmap(dl[['Gender', 'Age', 'Annual Income (k$)','Spending Score (1-100)']].corr(), annot = True)
plt.figure(1 , figsize = (15 , 5))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    sns.distplot(dl[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
plt.figure(1 , figsize = (15 , 15))

n = 0 

for cols in ['Age', 'Annual Income (k$)','Spending Score (1-100)']:

    n += 1 

    plt.subplot(1 , 3 , n)

    sns.violinplot(x = cols , y = 'Gender' , data = dl )

plt.show()
n = 0 

for x in [ 'Age', 'Annual Income (k$)','Spending Score (1-100)']:

    for y in [ 'Age', 'Annual Income (k$)','Spending Score (1-100)']:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = dl)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
X1=dl[['Age','Annual Income (k$)']]  

for gen in ['Female','Male']:

    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = dl[dl['Gender'] == gen] ,label = gen)

plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 

plt.legend()

plt.show()
X2=dl[['Age','Spending Score (1-100)']]  

for gen in ['Female','Male']:

    plt.scatter(x = 'Age' , y = 'Spending Score (1-100)' , data = dl[dl['Gender'] == gen] ,label = gen)

plt.xlabel('Age'), plt.ylabel('Spending Score (1-100)') 

plt.legend()

plt.show()
X3=dl[['Annual Income (k$)','Spending Score (1-100)']]  

for gen in ['Female','Male']:

    plt.scatter(x ='Annual Income (k$)' , y = 'Spending Score (1-100)' , data = dl[dl['Gender'] == gen] ,label = gen)

plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 

plt.legend()

plt.show()
kmeans = KMeans(n_clusters=4)

kmeans.fit(X2)

y_kmeans = kmeans.predict(X2)



plt.scatter(X2['Age'], X2['Spending Score (1-100)'], c=y_kmeans, s=50, cmap='viridis')

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)

kmeans.fit(X3)

y_kmeans = kmeans.predict(X3)



plt.scatter(X3['Annual Income (k$)'], X3['Spending Score (1-100)'], c=y_kmeans, s=50, cmap='viridis')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);