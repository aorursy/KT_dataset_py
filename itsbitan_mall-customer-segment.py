# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualsion

import seaborn as sns # data visualsion



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Impot the dataset

df = pd.read_csv('../input/Mall_Customers.csv')
#Lets a quick look of our dataset

df.info()
#Lets check the statistical inference of the dataset

df.describe()
#Lets check if any missing value in our dataset

df.isnull().sum()
#Count Plot of Gender

plt.figure(1 , figsize = (10 , 5))

sns.countplot(x = 'Gender' , data = df)

plt.show()
#Lets look the distribution of the Annual Income

sns.set(style = 'whitegrid')

sns.distplot(df['Annual Income (k$)'], color = 'blue')

plt.title('Distribution of Annual Income', fontsize = 20)

plt.xlabel('Range of Annual Income')

plt.ylabel('Count')
#Now look the distribution of Age

sns.set(style = 'whitegrid')

sns.distplot(df['Age'], color = 'red')

plt.title('Distribution of Age', fontsize = 20)

plt.xlabel('Range of Age')

plt.ylabel('Count')

plt.show()
#Now look the distribution of Spending Score

sns.set(style = 'whitegrid')

sns.distplot(df['Spending Score (1-100)'], color = 'green')

plt.title('Spending Score (1-100)', fontsize = 20)

plt.xlabel('Range of Spending Score (1-100)')

plt.ylabel('Count')

plt.show()
#Now look the correlation by ploting heatmap

corr = df.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (8,6))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.title('Correlation of df Features', y = 1.05, size=10)
#now look the pariplot of the dataset

sns.pairplot(df)

plt.title('Pairplot for the Data', fontsize = 20)

plt.show()
#Lets look the Age vs Annual Income

plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df) 

plt.xlabel('Age')

plt.ylabel('Annual Income (k$)')

plt.title('Age vs Annual Income w.r.t Gender')

plt.legend()

plt.show()

#Now look Age Vs Spending Score

plt.scatter(x = 'Age' , y = 'Spending Score (1-100)' , data = df) 

plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.title('Age vs Spending Score (1-100)')

plt.legend()

plt.show()
#Lets look the Annual Income vs Spending Score

plt.scatter(x = 'Annual Income (k$)' , y = 'Spending Score (1-100)' , data = df) 

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.title('Age vs Annual Income w.r.t Gender')

plt.legend()

plt.show()
#Lets take our matrice of features for ml model

x = df.iloc[:, [3, 4]].values
#Using the Elbow Method to find Optimal Number of Cluster

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    Kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    Kmeans.fit(x)

    wcss.append(Kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of Cluster')

plt.ylabel('WCSS')

plt.show() 
#Appling K-Means to the Mall Dataset

Kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init =10, random_state = 0)

y_Kmeans = Kmeans.fit_predict(x)  
#Visualsing the Cluster

plt.scatter(x[y_Kmeans == 0, 0], x[y_Kmeans == 0, 1], s = 50, c = 'red', label = 'Careful Customers group')

plt.scatter(x[y_Kmeans == 1, 0], x[y_Kmeans == 1, 1], s = 50, c = 'blue', label = 'Standard Customers group')

plt.scatter(x[y_Kmeans == 2, 0], x[y_Kmeans == 2, 1], s = 50, c = 'green', label = 'Target Customers group')

plt.scatter(x[y_Kmeans == 3, 0], x[y_Kmeans == 3, 1], s = 50, c = 'cyan', label = 'Careless Customers group')

plt.scatter(x[y_Kmeans == 4, 0], x[y_Kmeans == 4, 1], s = 50, c = 'magenta', label = 'Sensible Customers group')

plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], s = 100, c = 'Yellow', label = 'Centroid')

plt.title('Cluster of the Clients')

plt.xlabel('Annual Income(K$)')

plt.ylabel('Spending Scores(1-100)')

plt.legend()

plt.show()



#Using the Dendrogram to find optimal number of cluster

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

plt.title('Dendogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean Distance')

plt.show()

#Fitting Hierarchical Clustering to the Dataset

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(x)
#Visualising the Cluster

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 50, c = 'red', label = 'Careful Customers group')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 50, c = 'blue', label = 'Standard Customers group')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 50, c = 'green', label = 'Target Customers group')

plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 50, c = 'cyan', label = 'Careless Customers group')

plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 50, c = 'magenta', label = 'Sensible Customers group')

plt.title('Cluster of Customers')

plt.xlabel('Ananul Income(k$)')

plt.ylabel('Spending Score(1-100)')

plt.legend()

plt.show()
#Lets take our matrices of features

X = df.iloc[:, [2,4]].values 
#Using the Elbow Method to find Optimal Number of Cluster

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    Kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    Kmeans.fit(X)

    wcss.append(Kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of Cluster')

plt.ylabel('WCSS')

plt.show() 
#Appling K-Means to the Mall Dataset

Kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init =10, random_state = 0)

y_Kmeans = Kmeans.fit_predict(X)  
#Visualsing the Cluster

plt.scatter(X[y_Kmeans == 0, 0], X[y_Kmeans == 0, 1], s = 100, c = 'red', label = 'Customers should be paid more attention')

plt.scatter(X[y_Kmeans == 1, 0], X[y_Kmeans == 1, 1], s = 100, c = 'blue', label = 'Premium Customers group')

plt.scatter(X[y_Kmeans == 2, 0], X[y_Kmeans == 2, 1], s = 100, c = 'green', label = 'Customers have Potential ')

plt.scatter(X[y_Kmeans == 3, 0], X[y_Kmeans == 3, 1], s = 100, c = 'cyan', label = 'Customers should be treated carefully')

plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], s = 100, c = 'Yellow', label = 'Centroid')

plt.title('Cluster of the Clients')

plt.xlabel('Age')

plt.ylabel('Spending Scores(1-100)')

plt.legend()

plt.show()
#Using the Dendrogram to find optimal number of cluster

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean Distance')

plt.show()
#Fitting Hierarchical Clustering to the Dataset

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(X)
#Visualising the Cluster

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Customers should be treated carefully')

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Customers have Potential')

plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Customers should be paid more attention')

plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Premium Customers group')

plt.title('Cluster of Customers')

plt.xlabel('Age')

plt.ylabel('Spending Score(1-100)')

plt.legend(loc='upper right')

plt.show()