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
df["MINIMUM_PAYMENTS"]
df["MINIMUM_PAYMENTS"].median()
df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median(),inplace=True)
df["CREDIT_LIMIT"]
df["CREDIT_LIMIT"].median()
df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median(),inplace=True)
df.duplicated().sum()
df.drop(columns={'CUST_ID'}, inplace = True)

df.columns
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.heatmap(df.corr())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_df = scaler.fit_transform(df)

df_scaled = pd.DataFrame(scaled_df,columns=df.columns)

df_scaled.head()
from sklearn.cluster import KMeans

import pylab as pl

import random
k_means = KMeans(init = "k-means++", n_clusters = 8, n_init = 12)
k_means.fit(scaled_df)
k_means.labels_
k_means.cluster_centers_
Sum_of_squared_distances = []

K = range(1,21)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(df_scaled)

    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()