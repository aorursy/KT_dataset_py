import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sc

import matplotlib as mpl
data = pd.read_csv('/kaggle/input/marketing-data-for-a-supermarket-in-united-states/supermarket_marketing/Supermarket_CustomerMembers.csv')
data.head()
data.tail()
gender = {'Male': 0,'Female': 1}

data['Genre'] = [gender[item] for item in data['Genre']]
df = data.drop('CustomerID',axis=1)
df.head()
from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
scaled = pd.DataFrame(standard.fit_transform(df), columns = df.columns, index = df.index)
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.cluster import AgglomerativeClustering
Z = linkage(df, method = 'ward')

# plt.axhline(y =12.5, linestyle = '--')

dendrogram(Z, truncate_mode = 'lastp')

plt.xticks(rotation = 90)

plt.show()
ach = AgglomerativeClustering(n_clusters = 3)

ach.fit(df)
df['Labels'] = ach.labels_
df[df['Labels'] == 0].describe()
df[df['Labels'] == 1].describe()
df[df['Labels'] == 2].describe()
df['Labels'].value_counts()
sns.pairplot(df, hue='Labels')
from sklearn.cluster import KMeans
df = df.drop('Labels',axis =1)
inertia_list = []



for i in range(1,15):

    kmeans=KMeans(n_clusters = i)

    kmeans.fit(df)

    inertia_list.append(kmeans.inertia_)
plt.figure(figsize=(8,8))

plt.plot(range(1,15), inertia_list, color = 'blue', linestyle = 'dashed', marker ='o',markerfacecolor = 'red', markersize=10)

plt.title('Inertia vs K Value')

plt.xticks(range(1,15))

plt.xlabel('K')

plt.ylabel('Inertia')
kmeans = KMeans(n_clusters=6)
kmeans.fit(df)
centroids = kmeans.cluster_centers_
df['Labels'] = kmeans.labels_
df[df['Labels'] == 0].describe()
df[df['Labels'] == 1].describe()
df[df['Labels'] == 2].describe()
df[df['Labels'] == 3].describe()
df[df['Labels'] == 4].describe()
df[df['Labels'] == 5].describe()
df['Labels'].value_counts()
sns.pairplot(df,hue='Labels')
from sklearn.cluster import DBSCAN
df = df.drop('Labels',axis =1)
dbscan = DBSCAN().fit(df)
df['Labels'] = dbscan.labels_
df['Labels'].value_counts()
# from hdbscan import HDBSCAN
# df = df.drop('Labels',axis =1)
# hdbscan = HDBSCAN(min_cluster_size=3, min_samples=5, gen_min_span_tree=True)
# hdbscan.fit(df)
# df['Labels'] = hdbscan.labels_
# df[df['Labels'] == 0].describe()
# df[df['Labels'] == 1].describe()
# df[df['Labels'] == 2].describe()
# df[df['Labels'] == -1].describe()
# df['Labels'].value_counts()
# sns.pairplot(df,hue='Labels')