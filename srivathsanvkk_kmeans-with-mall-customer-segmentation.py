# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,MeanShift
data=pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')
data.head()
#Dropping Inefficient Feature

data.drop(['CustomerID'],axis=1,inplace=True)
data.head()
data.shape
#Encoding categorical feature to numerical for computation, Here only gender is categorical feature

from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()

data['Genre']=enc.fit_transform(data['Genre'])
#Male-1 and Female-0

data.head()
#Getting insights about the data and we can see age has some outliers since mean>median and income has some skewness

data.describe()
#Since we are clustering based on income and spending score so removing other columns

X = data.iloc[:, [2, 3]].values
X
#We can see there are more female percentage in the dataset

import seaborn as sns

sns.countplot(data['Genre'])
#we can see that income feature is right skewed.

sns.distplot(data['Annual Income (k$)'])
#Using elbow method to find what is best number of clusters

a = []

for i in range(1, 10):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 200, n_init = 10)

    kmeans.fit(X)

    a.append(kmeans.inertia_)
#List 

a
#Plotting the elbow method to find best number cluster

plt.plot(range(1,10),a)

plt.title('The Elbow Method',fontsize=20)

plt.xlabel('Number of clusters')

plt.ylabel(a)

plt.show()

#Using Kmeans cluster to get best outcome

kmeans=KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10)

pred=kmeans.fit_predict(X)

#Clustering based on income and spending

#Cluster1-rich and also spends a lot

#Cluster2-general(middle class)

#Cluster3-less income but spending a lot

#Cluster4-less income and spending less

#Cluster5-more income and spending very less (miser)

plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s = 100, c = 'cyan', label = 'Cluster1')

plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s = 100, c = 'blue', label = 'Cluster2')

plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s = 100, c = 'red', label = 'Cluster3')

plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s = 100, c = 'yellow', label = 'Cluster4')

plt.scatter(X[pred == 4, 0], X[pred == 4, 1], s = 100, c = 'black', label = 'Cluster5')

plt.title('Clusters of Customers')

plt.xlabel('Annual income(k$)')

plt.ylabel('spending score')

plt.legend()

plt.show()
#To cluster based on age and spending scores so removing other columns

X1=data.iloc[:, [1,3]].values
#Using Kmeans to cluster X1

pred1 = kmeans.fit_predict(X1)
#Cluster1-Middle Aged People Spending Average

#Cluster2-Youngsters spending lot

#Cluster3-People spread across age spending less

#Cluster4-Youngsters spending moderately

#Cluster5-Aged people Spending moderately





plt.scatter(X1[pred1== 0, 0], X1[pred1== 0, 1], s = 100, c = 'red', label = 'Cluster1')

plt.scatter(X1[pred1== 1, 0], X1[pred1== 1, 1], s = 100, c = 'blue', label = 'Cluster2')

plt.scatter(X1[pred1== 2, 0], X1[pred1== 2, 1], s = 100, c = 'green', label = 'Cluster3')

plt.scatter(X1[pred1== 3, 0], X1[pred1== 3, 1], s = 100, c = 'yellow', label = 'Cluster4')

plt.scatter(X1[pred1== 4, 0], X1[pred1== 4, 1], s = 100, c = 'pink', label = 'Cluster5')

plt.title('Clusters of Customers',)

plt.xlabel('Age')

plt.ylabel('Spending Scores')

plt.legend()

plt.show()
#Inorder to cluster based on age and annual income

X2=data.iloc[:, [1,2]].values
#Using Kmeans to cluster X1

pred2 = kmeans.fit_predict(X2)
#Cluster1-People spread until age 60 having moderate income

#Cluster2-People aged above 40 having less income indicating retired

#Cluster3-People spread across ages having less income

#Cluster4-Middle Aged people having very high income

#Cluster5-Youngsters having somewhat moderate income







plt.scatter(X2[pred2== 0, 0], X2[pred2== 0, 1], s = 100, c = 'red', label = 'Cluster1')

plt.scatter(X2[pred2== 1, 0], X2[pred2== 1, 1], s = 100, c = 'blue', label = 'Cluster2')

plt.scatter(X2[pred2== 2, 0], X2[pred2== 2, 1], s = 100, c = 'green', label = 'Cluster3')

plt.scatter(X2[pred2== 3, 0], X2[pred2== 3, 1], s = 100, c = 'yellow', label = 'Cluster4')

plt.scatter(X2[pred2== 4, 0], X2[pred2== 4, 1], s = 100, c = 'pink', label = 'Cluster5')

plt.title('Clusters of Customers',)

plt.xlabel('Age')

plt.ylabel('Annual Income')

plt.legend()

plt.show()
X3=data.iloc[:, [0,2]].values
#Using Kmeans to cluster X1

pred3 = kmeans.fit_predict(X3)
#Clustering based on gender and spending score

plt.scatter(X3[pred3== 0, 0], X3[pred3== 0, 1], s = 100, c = 'red', label = 'Cluster1')

plt.scatter(X3[pred3== 1, 0], X3[pred3== 1, 1], s = 100, c = 'blue', label = 'Cluster2')

plt.scatter(X3[pred3== 2, 0], X3[pred3== 2, 1], s = 100, c = 'green', label = 'Cluster3')

plt.scatter(X3[pred3== 3, 0], X3[pred3== 3, 1], s = 100, c = 'yellow', label = 'Cluster4')

plt.scatter(X3[pred3== 4, 0], X3[pred3== 4, 1], s = 100, c = 'pink', label = 'Cluster5')

plt.title('Clusters of Customers',)

plt.xlabel('Age')

plt.ylabel('Annual Income')

plt.legend()

plt.show()