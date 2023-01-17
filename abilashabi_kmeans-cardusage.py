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
import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/CreditCardUsage.csv")

df.head().T
df.shape
df.info()
df = df.drop("CUST_ID", axis=1)
df.columns
df.corr()
sns.heatmap(df.corr())
df.isna().sum()
df.MINIMUM_PAYMENTS.describe()
df.MINIMUM_PAYMENTS.median()
df.boxplot(column=['MINIMUM_PAYMENTS'])
df = df.fillna(df.median())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(df_scaled,columns= df.columns)

df_scaled.head()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []

for i in range(1,30):

    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300)

    kmeans.fit_predict(df_scaled)

    wcss.append(kmeans.inertia_)

plt.plot( wcss, 'ro-', label="WCSS")

plt.title("Computing WCSS for KMeans++")

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
kmeans = KMeans(n_clusters=8,init = "k-means++",n_init= 10,max_iter=300)

y_pred = kmeans.fit_predict( df_scaled )

labels = kmeans.labels_

df_scaled["Clus_km"] = labels

import seaborn as sns

df_scaled["cluster"] = y_pred

cols = list(df_scaled.columns)

sns.lmplot(data=df_scaled,x='BALANCE',y='PURCHASES',hue='Clus_km')

sns.set_palette('Set2')

sns.scatterplot(df_scaled['BALANCE'],df_scaled['PURCHASES'],hue=labels,palette='Set1')
x_dend = df_scaled

import scipy.cluster.hierarchy as sch



plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('Purchases')

plt.ylabel('Payments')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(x_dend, method = 'ward'))

plt.show()