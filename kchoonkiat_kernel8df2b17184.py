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
df = pd.read_csv('../input/Mall_Customers.csv')

df.head()
import seaborn as sns

%matplotlib inline

sns.heatmap(df.corr())
df.shape
df.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_tf = df

df_tf.Gender = le.fit_transform(df_tf['Gender'])
df_tf.head()
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
scaler = MinMaxScaler()
df_tf['Age'] = scaler.fit_transform(df_tf[['Age']])
df_tf['Annual Income (k$)'] = scaler.fit_transform(df_tf[['Annual Income (k$)']])

df_tf['Spending Score (1-100)'] = scaler.fit_transform(df_tf[['Spending Score (1-100)']])
df_tf.head()
X = df_tf.drop(['CustomerID','Spending Score (1-100)'], axis='columns')
X.head()
k_rng = range(1,10)

sse = []

for k in k_rng:

    km = KMeans(n_clusters=k)

    km.fit(X)

    sse.append(km.inertia_)
plt.plot(k_rng,sse)

plt.xlabel('K values')

plt.ylabel('Sum of Squared Errors')
km = KMeans(n_clusters=4)
y_4 = km.fit_predict(X)

y_4
X['cluster'] = y_4
X.head()
cluster0 = X.loc[X.cluster==0]

cluster1 = X.loc[X.cluster==1]

cluster2 = X.loc[X.cluster==2]

cluster3 = X.loc[X.cluster==3]
cluster1.head()
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot

import matplotlib.pyplot as plt

%matplotlib inline
fig = pyplot.figure()

ax = Axes3D(fig)

ax.set_xlabel('Gender')

ax.set_ylabel('Age')

ax.set_zlabel('Annual Income')

ax.scatter(cluster0['Gender'],cluster0['Age'],cluster0['Annual Income (k$)'],color='red')

ax.scatter(cluster1['Gender'],cluster1['Age'],cluster1['Annual Income (k$)'],color='blue')

ax.scatter(cluster2['Gender'],cluster2['Age'],cluster2['Annual Income (k$)'],color='green')

ax.scatter(cluster3['Gender'],cluster3['Age'],cluster3['Annual Income (k$)'],color='yellow')