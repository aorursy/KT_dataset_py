import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
df = pd.read_csv('../input/FullData.csv')

Names = pd.read_csv('../input/PlayerNames.csv')

df.assign(Index=0)

df['Index'] = [v.split('/')[2] for v in Names['url']]

df.head()
df2 = df[df.columns[19:54]].dropna()              #attributes only table - no need to normalise

                                                  #drop rows with NA values - otherwise clustering error

df3 = df2.copy()

del df3['Index']                                 #drop index values to run KMeans
df.Club_Position.unique()
kmeans = KMeans(n_clusters=6,n_init=100).fit(df3)     

kmeans.n_clusters
df2.assign(Cluster = np.nan)

df2['Cluster'] = kmeans.labels_

df = df.merge(df2, how="left")

df = df[df.Club_Position != 'Sub']

df = df[df.Club_Position != 'Res']                 #Res and Sub do not depend on attributes cluster

df = df.dropna(subset=['Club_Position'])
cluster_label = []

pos = df.Club_Position.unique()     #remove unwanted positions

for i in range(0, kmeans.n_clusters):

    pos_prop = []

    cluster_pos = df.loc[df['Cluster'] == i]['Club_Position'].tolist()    #find positions of players in the cluster

    for j in range(0, len(pos)):

        pos_prop.append(cluster_pos.count(pos[j])/df['Club_Position'].tolist().count(pos[j]))  #append proportion score

    top_prop= max(pos_prop)

    cluster_label.append([i, pos[pos_prop.index(top_prop)], top_prop])

cluster_label   #[cluster number, position, mode count]
