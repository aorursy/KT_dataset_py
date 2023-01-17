# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sklearn

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load data from source

players = pd.read_csv("../input/dota-2-matches/players.csv")

players.columns
#set column of data clustering

playersClustering = players[['hero_id', 'gold',

       'gold_per_min', 'xp_per_min', 'kills', 'deaths',

       'assists', 'hero_damage',

       'hero_healing', 'tower_damage', 'level', 'leaver_status']]
heroes = pd.read_csv('../input/dota-2-matches/hero_names.csv')

hero_lookup = dict(zip(heroes['hero_id'],heroes['localized_name']))

hero_lookup[0] = 'unknown'

playersClustering['hero'] = playersClustering['hero_id'].apply(lambda _id : hero_lookup[_id])
heroes_stats = playersClustering.groupby(['hero']).mean()

heroes_stats.drop('unknown',inplace=True)

print(heroes_stats)
heroes_clustering = heroes_stats[['gold_per_min','kills','deaths','assists','hero_damage','hero_healing','tower_damage']]

from sklearn.cluster import KMeans

n_clusters = 7

heroes_kmeans = KMeans(n_clusters=n_clusters,random_state=1000).fit(heroes_clustering)

# you need to set the random_State, otherwise the clustering numbering will keep changing
heroes_clustering['kmeans'] = heroes_kmeans.labels_
# generate group by stats that we set

kmeans_stats = heroes_clustering.groupby(['kmeans']).mean()

kmeans_stats['count'] = heroes_clustering.groupby(['kmeans'])['kills'].count()



# normalize

kmeans_statmeans = kmeans_stats.mean(axis=0)

kmeans_range = kmeans_stats.max(axis=0) - kmeans_stats.min(axis=0)

kmeans_statnorm = (kmeans_stats - kmeans_statmeans) / kmeans_range

kmeans_statnorm = kmeans_statnorm



# make plot

fig, (axis1, axis2) = plt.subplots(2,1,figsize=(14,14))

kmeans_stats['count'].plot.bar(ax=axis1)

kmeans_statnorm.iloc[:,:7].plot.bar(ax=axis2).legend(loc='lower left')

heroes_clustering['heroclass'] = 'na'

heroes_clustering.loc[(heroes_clustering['kmeans']==0),'heroclass8'] = 'Support'

heroes_clustering.loc[(heroes_clustering['kmeans']==1),'heroclass8'] = 'Any role'

heroes_clustering.loc[(heroes_clustering['kmeans']==2),'heroclass8'] = 'Powerful with hero'

heroes_clustering.loc[(heroes_clustering['kmeans']==3),'heroclass8'] = 'Ganker'

heroes_clustering.loc[(heroes_clustering['kmeans']==4),'heroclass8'] = 'Midlaner'

heroes_clustering.loc[(heroes_clustering['kmeans']==5),'heroclass8'] = 'Hard Support, Healer'

heroes_clustering.loc[(heroes_clustering['kmeans']==6),'heroclass8'] = 'Pusher'
heroes_clusteringSorted = heroes_clustering[['heroclass8','gold_per_min','kills','deaths','assists','hero_damage','hero_healing','tower_damage']].sort_values(['heroclass8'])

heroes_clusteringSorted.to_csv('heroCluster.csv',index=True)

heroes_clusteringSorted