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
df = pd.read_csv("../input/CreditCardUsage.csv")
df.head(5)
df.columns
df.describe().T
df.info()
df.isna().sum()
df["MINIMUM_PAYMENTS"].median()
df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median(),inplace=True)
df["CREDIT_LIMIT"].median()
df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median(),inplace=True)
df.duplicated().sum()
df.drop(columns={'CUST_ID'}, inplace = True)

df.columns
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(8,5))

plt.title("Balance distribution",fontsize=16)

plt.xlabel ("Balance",fontsize=14)

plt.grid(True)

plt.hist(df["BALANCE"],color='blue',edgecolor='k')

plt.show()
plt.figure(figsize=(10,7))

plt.ylabel("Balance",fontsize=18)

plt.xlabel("Purchase Frequency",fontsize=18)

plt.scatter(x=df["PURCHASES_FREQUENCY"],y=df["BALANCE"])
sns.boxplot(x=df["CREDIT_LIMIT"],y=df["BALANCE"])
sns.lineplot(x=df["CREDIT_LIMIT"],y=df["BALANCE"])
df.corr()["BALANCE"]
sns.heatmap(df.corr())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_df = scaler.fit_transform(df)

df_scaled = pd.DataFrame(scaled_df,columns=df.columns)

df_scaled.head()
from sklearn.cluster import KMeans

import pylab as pl

import random

import matplotlib.pyplot as plt
k_means = KMeans(init = "k-means++", n_clusters = 8, n_init = 12)
k_means.fit(scaled_df)
k_means_labels = k_means.labels_

k_means_labels
k_means_cluster_centers = k_means.cluster_centers_

k_means_cluster_centers
k_values = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in k_values]

kmeans

score = [kmeans[i].fit(scaled_df).score(scaled_df) for i in range(len(kmeans))]

score

pl.plot(k_values,score)

pl.xlabel('No. of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()
k_values = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in k_values]

kmeans

score = [kmeans[i].fit(scaled_df).inertia_ for i in range(len(kmeans))]

score

pl.plot(k_values,score)

plt.vlines(x=7,ymin=0,ymax=160000,linestyles='-')

pl.xlabel('Number of Clusters')

pl.ylabel('Sum of within sum square')

pl.title('Elbow Curve')

pl.show()