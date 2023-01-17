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
df = pd.read_csv("/kaggle/input/iris/Iris.csv")
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
import seaborn as sns

%matplotlib inline

import math

import matplotlib.pyplot as plt
sns.countplot(x='Species',data = df)
plt.scatter(df.PetalLengthCm,df.PetalWidthCm)

plt.xlabel("Petal Lenth In Cm")

plt.ylabel("Petal Width In Cm")

plt.legend()
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
km = KMeans(n_clusters = 2)

y_predicted = km.fit_predict(df[['PetalLengthCm','PetalWidthCm']])

y_predicted
df['Cluster'] = y_predicted

df = df.drop('Clauster',axis = 1)

df.head()
km.cluster_centers_
df1 = df[df.Cluster == 0]

df2 = df[df.Cluster == 1]
plt.scatter(df1.PetalLengthCm,df1.PetalWidthCm,color = 'green')

plt.scatter(df2.PetalLengthCm,df2.PetalWidthCm,color = 'red')

plt.xlabel("Petal Lenth In Cm")

plt.ylabel("Petal Width In Cm")

plt.legend()
scaler = MinMaxScaler()

scaler.fit(df[['PetalWidthCm']])

df['PetalWidthCm'] = scaler.transform(df[['PetalWidthCm']])

df.head()
scaler = MinMaxScaler()

scaler.fit(df[['PetalLengthCm']])

df['PetalLengthCm'] = scaler.transform(df[['PetalLengthCm']])

df.head()
km = KMeans(n_clusters = 2)

y_predicted = km.fit_predict(df[['PetalLengthCm','PetalWidthCm']])

y_predicted
df['Cluster'] = y_predicted

df.head()
km.cluster_centers_
df1 = df[df.Cluster == 0]

df2 = df[df.Cluster == 1]
plt.scatter(df1.PetalLengthCm,df1.PetalWidthCm,color = 'green')

plt.scatter(df2.PetalLengthCm,df2.PetalWidthCm,color = 'red')

plt.xlabel("Petal Lenth In Cm")

plt.ylabel("Petal Width In Cm")

plt.legend()

# for printing the centres

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'purple',marker = '*',label = 'Centroid')
# Sum of Square Error (sse)

sse = []

k_rng = range(1,10)

for k in k_rng:

    km = KMeans(n_clusters = k) 

    km.fit(df[['PetalLengthCm','PetalWidthCm']])

    sse.append(km.inertia_)      # It will find out the error in sse
plt.xlabel("k_Values")

plt.ylabel("Sum of Square Error")

plt.scatter(k_rng,sse)

plt.plot(k_rng,sse)