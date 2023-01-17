# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt # for adjusting graph

from sklearn.cluster import KMeans # clustering model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Import data

df = pd.read_csv('../input/Mall_Customers.csv')

df.head()
#Check the shape

df.shape
#Look at some simple statistics

df.describe()
#Check if there is null data and object type are correct

df.info()
#Histogram of Annual Income (k$)

sns.distplot(df['Annual Income (k$)'],kde = False,rug = True)

plt.title('Histogram of Annual Income (k$)')

plt.ylabel('Count')
sns.distplot(df['Spending Score (1-100)'],kde = False,rug = True)

plt.title('Histogram of Spending Score (1-100)')

plt.ylabel('Count')
sns.distplot(df['Age'],kde = False,rug = True)

plt.title('Histogram of Age')

plt.ylabel('Count')
#Plot each feature together in scatter plot

sns.pairplot(df.iloc[:,1:],kind='reg')
#Prepare data and convert catagorical data to dummies variable

x = df.iloc[:,1:]

x = pd.get_dummies(x)

x.head()
#Create model and fit with data

kmean = KMeans(n_clusters = 5 , random_state = 0).fit(x)

kmean.labels_
#Add the results back

df['Cluster'] = kmean.labels_ + 1

df.head()
#Groupby cluster and get average value from each cluster 

df.iloc[:,1:].groupby('Cluster').mean()