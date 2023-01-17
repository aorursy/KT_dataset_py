import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

%config InlineBackend.figure_format='retina'
data_size = 1000

df =pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df = df.sample(n=data_size)

df = df.fillna(0)

df.sample(5)
from sklearn import preprocessing

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch # draw dendrogram
Room_type = ['Entire home/apt','Private room','Shared room']

df['room_type'] = df['room_type'].replace(Room_type,['1','2','3']).to_numpy()
cols=['latitude', 'longitude', 'price', 'number_of_reviews','reviews_per_month','calculated_host_listings_count','room_type']
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)

mat = pt.fit_transform(df[cols])

mat[:5].round(4)
X=pd.DataFrame(mat, columns=cols)

X.tail()
df[cols].hist(layout=(1, len(cols)), figsize=(3*len(cols), 3.5));
X[cols].hist(layout=(1, len(cols)), figsize=(3*len(cols), 3.5), color='orange', alpha=.5);
Show_data = 30

fig, ax=plt.subplots(figsize=(20, 7))

dg=sch.dendrogram(sch.linkage(X[:Show_data], method='ward'), ax=ax, labels=df['name'][:Show_data].values)

# dg=sch.dendrogram(sch.linkage(df[cols], method='ward'), ax=ax)
sns.clustermap(X, col_cluster=False, cmap="Blues")
hc=AgglomerativeClustering(n_clusters=5, linkage='ward')

hc.fit(X)
df['cluster']=hc.labels_

df.head()
plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude', y='latitude', hue='cluster',s=20, data=df)