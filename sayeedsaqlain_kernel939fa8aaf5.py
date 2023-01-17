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
df = pd.read_csv('../input/Mall_Customers.csv')

df.head()
df.rename(columns={"Annual Income (k$)":"AIncome","Spending Score (1-100)":"Score"},inplace=True)
df.head()
df.shape
df.isna().sum()
sns.countplot(data=df, x='Gender')
plt.hist(data=df,x='Age',bins=[10,20,30,40,50,60,70,80],color='Green')

plt.xlabel('Age')
plt.hist(data=df,x='AIncome',bins=[10,20,30,40,50,60,70,80,90,100,110,120,130,140],color='Grey')
X = df.drop(columns=['CustomerID', 'Gender', 'AIncome'])

X.head()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4).fit(X)

labels = kmeans.labels_

centroids = kmeans.cluster_centers_
x=df['Age']

y=df['Score']



plt.scatter(x,y,c=labels)

plt.scatter(centroids[:,0],centroids[:,1],color='red')

plt.xlabel('Age')

plt.ylabel('Spending Score')
X2 = df.drop(columns=['CustomerID','Gender','Age'])

X2.head()
kmeans2 = KMeans(n_clusters=4).fit(X2)

labels2 = kmeans2.labels_

centroid2 = kmeans2.cluster_centers_
x2 = df['AIncome']

y2 = df['Score']



plt.scatter(x2,y2,c=labels2)

plt.scatter(centroid2[:,0],centroid2[:,1],color='red')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')