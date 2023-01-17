# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Mall_Customers.csv")

data.head()
data.tail()
data.isnull().sum()
data.describe()
data.drop(['CustomerID'], axis = 1, inplace = True)
data['Gender'] = [0 if x == "Female" else 1 for x in data.Gender]

data.head()
# Data visualization

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize = (6,6))

plt.subplot(212)

plt.title("Gender-wise distribution: 0 - Female, 1 - Male")

plt.pie(data['Gender'].value_counts(sort = True), labels = np.unique(data.Gender), autopct='%1.1f%%')
plt.figure(figsize = (12,12))

plt.subplot(212)

plt.title("Annual income vs. Spending score")

plt.plot(data["Annual Income (k$)"], data["Spending Score (1-100)"])

plt.show()
from sklearn.cluster import KMeans as km

import tensorflow as tf



#Elbow method to find optimal value of K

#Source - 

kvals = []

for i in range(1,10):

    kmeans = km(n_clusters = i, init = "k-means++", random_state = 0)

    kmeans.fit(data)

    kvals.append(kmeans.inertia_)
#Visualizing the ELBOW method to get the optimal value of K 

plt.plot(range(1,10), kvals)

plt.title('The Elbow Method')

plt.xlabel('no. of clusters')

plt.ylabel('kvals')

plt.show()
X= data.iloc[:, [2,3]].values

kmm = km(n_clusters = 6,random_state = 0)

kmm.fit(X)

y_kmeans = kmm.predict(X)



plt.figure(figsize = (6,6))

plt.scatter(X[:, 0], X[:, 1], c = y_kmeans)



centers = kmm.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='red')

plt.title('Segments of customers')

plt.xlabel('Annual Income (k$) ->')

plt.ylabel('Spending Score (1-100) ->')
