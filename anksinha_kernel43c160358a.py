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

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

#Import the data

df=pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

df.head()

df.describe()

print(df.shape)
#Plot the data for Annual Income and Spending Score

plt.figure(figsize=(6,6))

plt.scatter(df.iloc[:,3],df.iloc[:,4])

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('Visualization of raw data for Mall Spending-Annual Income vs Spending Score')
#Plot the data for Age and Spending Score

plt.figure(figsize=(6,6))

plt.scatter(df.iloc[:,2],df.iloc[:,4])

plt.xlabel('Age')

plt.ylabel('Spending Score')

plt.title('Visualization of raw data for Mall Spending - Age vs Spending Score')

#Clustering of Data 

#Drop columns CustomerID, Gender and Age for a pure Annual Income vs Spending Score segmentation.

#Standardize the data

df_inc_spend = df.drop(['CustomerID','Gender','Age'], axis=1)

df_inc_spend.head()

x_std =StandardScaler().fit_transform(df_inc_spend)

km = KMeans(n_clusters=2, max_iter=100)

km.fit(x_std)

centroids = km.cluster_centers_

#Plot the clustered data

fig, ax = plt.subplots(figsize=(8, 8))

plt.scatter(x_std[km.labels_ == 0, 0], x_std[km.labels_ == 0, 1],

            c='green', label='cluster 1')

plt.scatter(x_std[km.labels_ == 1, 0], x_std[km.labels_ == 1, 1],

            c='blue', label='cluster 2')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,

            c='r', label='centroid')

plt.legend()

#plt.xlim([-2, 2])

#plt.ylim([-2, 2])

plt.xlabel('Income')

plt.ylabel('Score')

plt.title('Visualization of clustered data', fontweight='bold')

ax.set_aspect('equal');

#Clustering of Data 

#Drop columns CustomerID, Gender and Age for a pure Annual Income vs Spending Score segmentation.

#Standardize the data

df_age_spend = df.drop(['CustomerID','Gender','Annual Income (k$)'], axis=1)

df_age_spend.head()

x_std =StandardScaler().fit_transform(df_age_spend)

km = KMeans(n_clusters=2, max_iter=100)

km.fit(x_std)

centroids = km.cluster_centers_

#Plot the clustered data

fig, ax = plt.subplots(figsize=(8, 8))

plt.scatter(x_std[km.labels_ == 0, 0], x_std[km.labels_ == 0, 1],

            c='green', label='cluster 1')

plt.scatter(x_std[km.labels_ == 1, 0], x_std[km.labels_ == 1, 1],

            c='blue', label='cluster 2')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,

            c='r', label='centroid')

plt.legend()

#plt.xlim([-2, 2])

#plt.ylim([-2, 2])

plt.xlabel('Age')

plt.ylabel('Spendin Score')

plt.title('Visualization of clustered data', fontweight='bold')

ax.set_aspect('equal');
