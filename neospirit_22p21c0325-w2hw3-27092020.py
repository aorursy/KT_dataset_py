!pip install fastcluster
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")



df.head()
print(df.isna().sum())
df = df.sample(n=1000)

df = df.copy().dropna(subset=["name", "host_name"])

df = df.fillna(0)
from sklearn import preprocessing

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch # draw dendrogram
Room_type = ['Private room','Entire home/apt','Shared room']

df['room_type'] = df['room_type'].replace(Room_type,['1','2','3']).to_numpy()



cols=['latitude', 'longitude', 'price', 'number_of_reviews','reviews_per_month','calculated_host_listings_count','room_type']
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)

mt = pt.fit_transform(df[cols]).round(3)



X = pd.DataFrame(mt, columns=cols)
X
Show_data = 100

fig, ax=plt.subplots(figsize=(20, 7))

dg=sch.dendrogram(sch.linkage(X[:Show_data], method='ward'), ax=ax, labels=df['name'][:Show_data].values)

# dg=sch.dendrogram(sch.linkage(df[cols], method='ward'), ax=ax)
g = sns.clustermap(X, metric="correlation", cmap="mako")
hc=AgglomerativeClustering(n_clusters=5, linkage='ward')

hc.fit(X)



sns.scatterplot(x='longitude', y='latitude',s=20, data=df)