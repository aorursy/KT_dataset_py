# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Other libraries we need

import sqlite3
with sqlite3.connect('../input/soccer/database.sqlite') as con:

    player_attributes = pd.read_sql_query("SELECT * from Player_Attributes",con)

    team = pd.read_sql_query("SELECT * from Team", con)

    match = pd.read_sql_query("SELECT * from Match", con)

    player = pd.read_sql_query("SELECT * from Player",con)

    league = pd.read_sql_query("SELECT * from League", con)

    country = pd.read_sql_query("SELECT * from Country", con)

    team_attributes = pd.read_sql_query("SELECT * from Team_Attributes",con)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.options.display.max_columns = 250
player_attributes.head()
player.head()

player.info()
team.head()
match.head()
league.head()
country.head()
team_attributes.head()
player_att_max_date = player_attributes.groupby(["player_api_id"]).date.max() 

player_att_data = pd.merge(player_att_max_date, player_attributes, on= ('player_api_id', 'date'), how= 'left')

player_att_data.head()
player_data = pd.merge(player, player_att_data, on= 'player_api_id', how= 'left')

player_data.info()
def position(x):

    if (x['attacking_work_rate'] == 'high') & (x['defensive_work_rate'] == 'low'):

        return 'striker'

    if (x['attacking_work_rate'] == 'high') & (x['defensive_work_rate'] == 'medium'):

        return 'number10'

    elif (x['attacking_work_rate'] == 'low') & (x['defensive_work_rate'] == 'high'):

        return 'defencer'

    elif (x['attacking_work_rate'] == 'medium') & (x['defensive_work_rate'] == 'high'):

        return 'number6'

    elif (x['attacking_work_rate'] == 'medium') & (x['defensive_work_rate'] == 'medium'):

        return 'midfielder'

    else :

        return 'unknown'

    return 
player_data['player_position']= player_data.apply(lambda x: position(x), axis= 1)

player_data.head()
player_data['player_position'].value_counts()
player_data= player_data[player_data['player_position'] != 'unknown']

player_data['player_position'].value_counts()
player_data_analyze= player_data.iloc[:,np.r_[2, 12, 15:43]]
player_data_analyze.dropna(subset=['preferred_foot'], inplace= True)

player_data_analyze.fillna(0, inplace= True)

player_data_analyze.info()
player_data_analyze.preferred_foot= player_data.preferred_foot.replace('right', 1).replace('left', 0)

player_data_analyze
player_data_analyze.crossing= player_data_analyze.crossing / 100

player_data_analyze.finishing= player_data_analyze.finishing / 100

player_data_analyze.heading_accuracy	= player_data_analyze.heading_accuracy / 100

player_data_analyze.short_passing= player_data_analyze.short_passing / 100

player_data_analyze.volleys= player_data_analyze.volleys / 100

player_data_analyze.dribbling= player_data_analyze.dribbling / 100

player_data_analyze.curve= player_data_analyze.curve / 100

player_data_analyze.free_kick_accuracy= player_data_analyze.free_kick_accuracy / 100

player_data_analyze.long_passing= player_data_analyze.long_passing / 100

player_data_analyze.ball_control= player_data_analyze.ball_control / 100

player_data_analyze.acceleration= player_data_analyze.acceleration / 100

player_data_analyze.sprint_speed= player_data_analyze.sprint_speed / 100

player_data_analyze.agility= player_data_analyze.agility / 100

player_data_analyze.reactions= player_data_analyze.reactions / 100

player_data_analyze.balance= player_data_analyze.balance / 100

player_data_analyze.shot_power= player_data_analyze.shot_power / 100

player_data_analyze.jumping= player_data_analyze.jumping / 100

player_data_analyze.stamina= player_data_analyze.stamina / 100

player_data_analyze.strength= player_data_analyze.strength / 100

player_data_analyze.long_shots= player_data_analyze.long_shots / 100

player_data_analyze.aggression= player_data_analyze.aggression / 100

player_data_analyze.interceptions= player_data_analyze.interceptions / 100

player_data_analyze.positioning= player_data_analyze.positioning / 100

player_data_analyze.vision= player_data_analyze.vision / 100

player_data_analyze.penalties= player_data_analyze.penalties / 100

player_data_analyze.marking= player_data_analyze.marking / 100

player_data_analyze.standing_tackle= player_data_analyze.standing_tackle / 100

player_data_analyze.sliding_tackle= player_data_analyze.sliding_tackle / 100



player_data_analyze
player_data_analyze.corr()
import seaborn as sns

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(25,25))

heat_map = sns.heatmap(player_data_analyze.corr(), vmin = 0, vmax= 1, annot= True, fmt='.1f' , linewidths=.5, ax= ax)

plt.show()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)

player_kmeans_data = player_data_analyze.iloc[:,1:]

kmeans.fit(player_kmeans_data)

y_kmeans = kmeans.predict(player_kmeans_data)
plt.scatter(player_kmeans_data.loc[:, 'marking'], player_kmeans_data.loc[:, 'shot_power'], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
y_kmeans

type(y_kmeans)
player_data['player_position'].value_counts()
unique_elements, counts_elements = np.unique(y_kmeans, return_counts=True)

unique_elements

counts_elements
player_data['player_cluster']= y_kmeans.tolist()

player_data.head()
player_data.groupby(['player_cluster', 'player_position']).player_name.count()
from sklearn.decomposition import PCA

pca= PCA(n_components= 3, whiten= True)

pca.fit(player_kmeans_data)

pca_analyze= pca.transform(player_kmeans_data)

print("PCA Explained Variation Ratio", pca.explained_variance_ratio_)

print("Sum of Variation Ratio", sum(pca.explained_variance_ratio_))
kmeans = KMeans(n_clusters=5)

kmeans.fit(pca_analyze)

pca_y_kmeans = kmeans.predict(pca_analyze)
player_data['player_position'].value_counts()

unique_elements, counts_elements = np.unique(pca_y_kmeans, return_counts=True)

unique_elements

counts_elements
player_data['player_cluster_pca']= pca_y_kmeans.tolist()

player_data.groupby(['player_position', 'player_cluster']).player_name.count()

player_data.groupby(['player_position', 'player_cluster_pca']).player_name.count()
from sklearn.metrics import silhouette_samples, silhouette_score
silhouette_avg= silhouette_score(pca_analyze, pca_y_kmeans)
kmeans4 = KMeans(n_clusters=4)

kmeans4.fit(pca_analyze)

pca_y4_kmeans = kmeans4.predict(pca_analyze)

silhouette_avg4= silhouette_score(pca_analyze, pca_y4_kmeans)
kmeans3 = KMeans(n_clusters=3)

kmeans3.fit(pca_analyze)

pca_y3_kmeans = kmeans3.predict(pca_analyze)

silhouette_avg3= silhouette_score(pca_analyze, pca_y3_kmeans)
kmeans2 = KMeans(n_clusters=2)

kmeans2.fit(pca_analyze)

pca_y2_kmeans = kmeans2.predict(pca_analyze)

silhouette_avg2= silhouette_score(pca_analyze, pca_y2_kmeans)
kmeans6 = KMeans(n_clusters=6)

kmeans6.fit(pca_analyze)

pca_y6_kmeans = kmeans6.predict(pca_analyze)

silhouette_avg6= silhouette_score(pca_analyze, pca_y6_kmeans)
kmeans7 = KMeans(n_clusters=7)

kmeans7.fit(pca_analyze)

pca_y7_kmeans = kmeans2.predict(pca_analyze)

silhouette_avg7= silhouette_score(pca_analyze, pca_y7_kmeans)

print("For 2 clusters, the average silhouette score is ", silhouette_avg2)

print("For 3 clusters, the average silhouette score is ", silhouette_avg3)

print("For 4 clusters, the average silhouette score is ", silhouette_avg4)

print("For 5 clusters, the average silhouette score is ", silhouette_avg)

print("For 6 clusters, the average silhouette score is ", silhouette_avg6)

print("For 7 clusters, the average silhouette score is ", silhouette_avg7)