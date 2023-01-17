# Load Python libraries

import re

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

from itertools import compress

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

plt.style.use('fivethirtyeight')
# Load data

df = pd.read_csv('../input/results_by_booth_2015 - english - v3.csv', encoding='iso-8859-1')



# Select columns

df = df.drop(df.columns[[0,1,3,4,5,6,7]], axis=1)
df.head()
# The biggest settlements

df = df.groupby('settlement_name_english').sum()

settl_top = df.sum(axis=1).reset_index()

settl_top.columns = ['settlement', 'values']

settl_top = settl_top.sort_values('values', ascending=False)

settl_top['percent'] = round(settl_top['values']/settl_top['values'].sum()*100, 2)
# Plot "The biggest israili settlements"

fig = plt.figure(figsize=(11, 5))

ax = fig.add_subplot(111)

sns.barplot(y='settlement', x='values', data=settl_top.head(20))

plt.title('The biggest israili settlements (top-20)')

plt.ylabel('Settlements')

plt.xlabel('% of all votes')

plt.show()
# Top-10 parties

party_top = df.sum().reset_index()

party_top.columns = ['party', 'values']

party_top = party_top.sort_values('values', ascending=False)

party_top['percent'] = round(party_top['values']/party_top['values'].sum()*100, 2)
# Plot "The most popular israili parties in 2015"

fig = plt.figure(figsize=(11, 5))

ax = fig.add_subplot(111)

sns.barplot(y='party', x='percent', data=party_top.head(15))

plt.title('The most popular israili parties in 2015 (top-15)')

plt.ylabel('Parties')

plt.xlabel('% of all votes')

plt.show()
# Optimal number of clusters

distortions = []

K = range(1,30)

for k in K:

    kmeanModel = KMeans(n_clusters=k, max_iter=10000).fit(df)

    kmeanModel.fit(df)

    distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

plt.figure(figsize=(11, 5))

plt.plot(K, distortions, 'o-', markersize=12)

plt.xlabel('Number of clusters')

plt.ylabel('Distortion')

plt.title('Optimal number of clusters -- 7')

plt.show()
# Сluster analysis (k-means)

kmeanModel = KMeans(n_clusters=7, max_iter=10000).fit(df)

df_clusts = pd.DataFrame({'ID': df.index, 'clusts': kmeanModel.labels_})



# Add Clusters to DF

df = pd.merge(df, df_clusts, right_on='ID', left_index=True)
# Add top paties in each cluster

data = df.drop('ID', axis=1)

data = data.groupby('clusts').sum()

for i in data.index:

    data.iloc[i,:] = round(data.iloc[i,:]/data.iloc[i,:].sum()*100, 2).tolist()

data1 = pd.DataFrame({'clusts': data.index,

                     'party': str(data.index)})

for i in data.index:

    reg = re.sub(' {2,}', ' - ', str(data.iloc[i,:].sort_values(ascending=False)[0:5]))

    reg = re.sub('\n', ', ', reg)

    reg = re.sub(', Name:.*', '', reg)

    data1.iloc[i,1] = reg

data = data1

del data1



# Add cities from top-50 in each clust

data['cities'] = 'text'

settles = df[['ID', 'clusts']]

settles.index = settles.pop('ID')

settles = settles.filter(items=settl_top.head(50).settlement.tolist(), axis=0).reset_index()

for i in set(data.clusts):

    reg = ', '.join(settles.ID[settles.clusts == i])

    data.ix[i,'cities'] = reg



# Add number of all cities in each cluster

data = pd.merge(data, df.groupby('clusts')['ID'].count().reset_index())

data = data.drop('clusts', axis=1)

data.columns = ['Top-5 parties by % of votes in the cluster', 

                'The biggest (of The Top-50) cities in the cluster',

                'All cities in the clust (amount)']
data