# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
customers = pd.read_csv("../input/Mall_Customers.csv")

print('Data shape\n ', customers.shape)

print('Data Info', customers.info());
customers.head()
del customers['CustomerID']

customers.describe().T
customers.head()
customers.nunique()
sns.pairplot(customers);
customers.groupby('Gender')['Annual Income (k$)'].sum().plot.bar();
plt.scatter(x=customers['Annual Income (k$)'], y=customers['Spending Score (1-100)'], c=customers['Age'])
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler



df = customers.iloc[:, 1:-1]

scale = StandardScaler()

scale.fit(df)

X_scaled = scale.transform(df)
#best number of k using elbow method

k =[]

for i in range(1,11):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(X_scaled)

    k.append(kmeans.inertia_)



plt.figure(figsize=(10,5))

plt.plot(range(1,11), k)

plt.title('Elbow Method')

plt.xlabel('Cluster Number')

plt.ylabel('Value');
Kmeans = KMeans(n_clusters= 3)

Kmeans.fit(X_scaled)

cluster = Kmeans.predict(X_scaled)

plt.figure(figsize=(12,8))

plt.scatter(x=customers['Annual Income (k$)'], y=customers['Spending Score (1-100)'], c=cluster)

# #plt.scatter(x=Kmeans.cluster_centers_[:,0], y= Kmeans.cluster_centers_[:,1], color='black', marker='*', )

# plt.title('5 Costumer Cluster')

# plt.xlabel('Annual Income')

# plt.ylabel('Spending Score');

# plt.grid(True)