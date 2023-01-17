# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



%matplotlib inline

import seaborn as sns

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/CreditCardUsage.csv")

df.head()
sns.scatterplot(x=df['BALANCE'], y=df['PURCHASES'],hue=df['BALANCE_FREQUENCY'])
df.info()
df.duplicated().sum()
df = df.drop('CUST_ID', axis=1)

df.head()
df.shape
df.isna().sum()
#df.MINIMUM_PAYMENTS=df.MINIMUM_PAYMENTS.dropna()

#df.loc[df['MINIMUM_PAYMENTS'].isna()]

df.drop(df.loc[df['MINIMUM_PAYMENTS'].isna()].index, inplace=True)
df.drop(df.loc[df['CREDIT_LIMIT'].isna()].index, inplace=True)
from sklearn.preprocessing import StandardScaler

X = df.values[:,1:]

X = np.nan_to_num(X)

Clus_dataSet = StandardScaler().fit_transform(X)

Clus_dataSet
from sklearn.cluster import KMeans

import pylab as pl

%matplotlib inline

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

score

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()
clusterNum = 3

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(X)

labels = k_means.labels_

print(labels)
df["Clus_km"] = labels

df.head(5)
df_GroupValue=df.groupby('Clus_km').mean()

df_GroupValue
df.info()
fig, ax = plt.subplots(figsize=(15,7))

df_GroupValue.plot(ax=ax)
fig, ax = plt.subplots(figsize=(15,7))

df_GroupValue[['BALANCE_FREQUENCY','BALANCE']].plot(ax=ax)
fig, ax = plt.subplots(figsize=(15,7))

df_GroupValue[['PURCHASES_FREQUENCY','PURCHASES','CREDIT_LIMIT']].plot(ax=ax)
#fig, ax = plt.subplots(figsize=(15,7))

#df.groupby('Clus_km').mean()

df.plot(x='Clus_km',y='PAYMENTS')
sns.scatterplot(x=df['Clus_km'], y=df['PAYMENTS'],hue=df['PURCHASES'])