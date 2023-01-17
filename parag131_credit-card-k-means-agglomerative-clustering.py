# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/ccdata/CC GENERAL.csv')

data.head()
data.shape
data.describe()
data.isnull().sum()
data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean(),inplace=True)

data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean(),inplace=True)
data.isnull().sum()
data.columns
data.drop('CUST_ID',axis=1,inplace=True)
data.columns
from scipy.stats import zscore
data_scaled=data.apply(zscore)

data_scaled.head()
cluster_range = range(1,15)

cluster_errors=[]

for i in cluster_range:

    clusters=KMeans(i)

    clusters.fit(data_scaled)

    labels=clusters.labels_

    centroids=clusters.cluster_centers_,3

    cluster_errors.append(clusters.inertia_)

clusters_df=pd.DataFrame({'num_clusters':cluster_range,'cluster_errors':cluster_errors})

clusters_df
f,ax=plt.subplots(figsize=(15,6))

plt.plot(clusters_df.num_clusters,clusters_df.cluster_errors,marker='o')

plt.show()
kmean= KMeans(4)

kmean.fit(data_scaled)

labels=kmean.labels_
clusters=pd.concat([data, pd.DataFrame({'cluster':labels})], axis=1)

clusters.head()
for c in clusters:

    grid= sns.FacetGrid(clusters, col='cluster')

    grid.map(plt.hist, c)
clusters.groupby('cluster').mean()
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(data_scaled)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])

principalDf.head(2)
finalDf = pd.concat([principalDf, pd.DataFrame({'cluster':labels})], axis = 1)

finalDf.head()


plt.figure(figsize=(15,10))

ax = sns.scatterplot(x="principal component 1", y="principal component 2", hue="cluster", data=finalDf,palette=['red','blue','green','yellow'])

plt.show()
data_scaled.head()
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram,linkage
Z=linkage(data_scaled,method="ward")
plt.figure(figsize=(15,10))

dendrogram(Z,leaf_rotation=90,p=5,color_threshold=20,leaf_font_size=10,truncate_mode='level')

plt.axhline(y=125, color='r', linestyle='--')

plt.show()
model=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
model.fit(data_scaled)
model.labels_
clusters_agg=pd.concat([data, pd.DataFrame({'cluster':model.labels_})], axis=1)

clusters_agg.head()
clusters_agg.groupby('cluster').mean()
finalDf_agg = pd.concat([principalDf, pd.DataFrame({'cluster':model.labels_})], axis = 1)

finalDf_agg.head()
plt.figure(figsize=(15,10))

ax = sns.scatterplot(x="principal component 1", y="principal component 2", hue="cluster", data=finalDf_agg,palette=['red','blue','green','yellow'])

plt.show()