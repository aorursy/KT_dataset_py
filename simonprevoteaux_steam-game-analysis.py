from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import re

import seaborn as sns

from sklearn.cluster import KMeans

df = pd.read_csv("../input/steam-200k.csv", header=None, index_col=None, names=['UserID', 'Game', 'Action', 'Hours', 'Other'])

df.head()
# infos for action "play", seems more relevant to me.

# we want here to exclude the action = 'purchase' with Hours = 1.0 that are not relevant in the mean computation 

df.loc[df['Action'] == 'play'].describe()



values = df.groupby(['UserID', 'Action']).size()

values.head()
print("Number of games : {0}".format(len(df.Game.unique())))

print("Number of users : {0}".format(len(df.UserID.unique())))

print("Number of total purchases : {0}".format(len(df.loc[df['Action'] == 'purchase'])))

print("Number of total plays infos : {0}".format(len(df.loc[df['Action'] == 'play'])))







nb_games = 10

df_purchase = df.loc[df['Action'] == 'purchase']

purchased_times = df_purchase.groupby('Game')['Game'].agg('count').sort_values(ascending=False)

purchased_times = pd.DataFrame({'game': purchased_times.index, 'times_purchased': purchased_times.values})[0:nb_games]



df_play = df.loc[df['Action'] == 'play']

hours_played = df_play.groupby('Game')['Hours'].agg(np.sum).sort_values(ascending=False)

hours_played = pd.DataFrame({'game': hours_played.index, 'hours_played': hours_played.values})[0:nb_games]



fig, ax =plt.subplots(1,2,figsize=(12,nb_games))



sns.barplot(y = 'game', x = 'times_purchased', data = purchased_times, ax=ax[0])

sns.barplot(y = 'game', x = 'hours_played', data = hours_played, ax=ax[1])





ax[1].yaxis.tick_right()

ax[1].yaxis.set_label_position("right")

for i in range(0,2):

    ax[i].tick_params(axis='y', labelsize=18)

    ax[i].xaxis.label.set_size(20)

top = 10

user_counts = df.groupby('UserID')['Hours'].agg(np.sum).sort_values(ascending=False)[0:top]

mask = df['UserID'].isin(user_counts.index)

df_infos_user = df.loc[mask].loc[df['Action'] == 'play']

hours_played = df_infos_user.groupby('Game')['Hours'].agg(np.sum).sort_values(ascending=False)

hours_played = pd.DataFrame({'game': hours_played.index, 'hours_played': hours_played.values})[0:nb_games]



sns.barplot(y = 'game', x = 'hours_played', data = hours_played)



nb_top_games = 20

hours_played = df_play.groupby('Game')['Hours'].agg(np.sum).sort_values(ascending=False)

top_played_games = pd.DataFrame({'game': hours_played.index, 'hours_played': hours_played.values})[0:nb_top_games]



mask = df['Game'].isin(top_played_games['game'])



df_infos_user = df.loc[mask].loc[df['Action'] == 'play'][['Hours', 'Game']]





sns.set_style("whitegrid")

sns.boxplot(x="Hours", y="Game", data=df_infos_user, palette="Set3")

df_purchased_games = df.loc[df['Action'] == 'purchase']

df_played_games = df.loc[df['Action'] == 'play']



#here we compute the number of games a user has bought

user_counts = df_purchased_games.groupby('UserID')['UserID'].agg('count').sort_values(ascending=False)

#here we compute the number of hours he has played 

hours_played = df_played_games.groupby('UserID')['Hours'].agg(np.sum).sort_values(ascending=False)



#df creation

user_df_purchased_games = pd.DataFrame({'UserID': user_counts.index, 'nb_purchased_games': user_counts.values})

user_df_hours_played = pd.DataFrame({'UserID': hours_played.index, 'hours_played': hours_played.values})



#merge to have one entry per user with number of hours played and number of purchased games

data = pd.merge(user_df_purchased_games, user_df_hours_played, on='UserID')

sns.jointplot(x="nb_purchased_games", y="hours_played", data=data)# , kind="reg")
g = sns.jointplot(x="nb_purchased_games", y="hours_played", data=data )#, kind="reg")

ax = g.ax_joint

ax.set_yscale('log')

g.ax_marg_y.set_yscale('log')

g
#here we compute the number of games a user has played

user_counts = df_played_games.groupby('UserID')['UserID'].agg('count').sort_values(ascending=False)

#here we compute the number of hours he has played 

hours_played = df_played_games.groupby('UserID')['Hours'].agg(np.sum).sort_values(ascending=False)



#df creation

user_df_played_games = pd.DataFrame({'UserID': user_counts.index, 'nb_played_games': user_counts.values})

user_df_hours_played = pd.DataFrame({'UserID': hours_played.index, 'hours_played': hours_played.values})





#merge to have one entry per user with number of hours played and number of played games

data = pd.merge(user_df_played_games, user_df_hours_played, on='UserID')



sns.jointplot(x="nb_played_games", y="hours_played", data=data )# , kind="reg")

g = sns.jointplot(x="nb_played_games", y="hours_played", data=data )#, kind="reg")

ax = g.ax_joint

ax.set_yscale('log')

g.ax_marg_y.set_yscale('log')

g
temp = pd.merge(user_df_purchased_games, data, on='UserID')

temp = temp.copy()

del temp['UserID'] #don't need this for k mean

sns.heatmap(temp.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlatio
# K Means

temp = data.copy()

del temp['UserID'] #don't need this for k mean



N_CLUSTERS = 6

train_data = temp.as_matrix()

# Using sklearn

km = KMeans(n_clusters=N_CLUSTERS)

km.fit(train_data)

# Get cluster assignment labels

labels = km.labels_ # 0 to n_clusters-1

# Format results as a DataFrame



LABEL_COLOR_MAP = {0 : 'red',

                   1 : 'blue',

                   2 : 'green',

                   3 : 'yellow',

                   4 : 'orange',

                   5 : 'pink'

                  }



label_color = [LABEL_COLOR_MAP[l] for l in labels]





fig, ax =plt.subplots(2,1,figsize=(10,10))

ax[0].scatter(data[['nb_played_games']], data[['hours_played']], c=label_color, s=8, marker='o')

ax[0].set_xlabel('nb_played_games')

ax[0].set_ylabel('hours_played')

ax[0].set_title('K Means')



ax[1].scatter(data[['nb_played_games']], data[['hours_played']], c=label_color, s=8, marker='o')

ax[1].set_xlabel('nb_played_games')

ax[1].set_ylabel('hours_played(log)')

ax[1].set_title('K Means with y log scale')

ax[1].set_yscale('log')

from sklearn.mixture import GaussianMixture

cov_types = ['spherical', 'diag', 'tied', 'full']



gm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag')

gm.fit(train_data)

y_train_pred = gm.predict(train_data)

label_color = [LABEL_COLOR_MAP[l] for l in y_train_pred]



fig, ax =plt.subplots(3,1,figsize=(10,10))

ax[0].scatter(data[['nb_played_games']], data[['hours_played']], c=label_color, s=8, marker='o')

ax[0].set_xlabel('nb_played_games')

ax[0].set_ylabel('hours_played')

ax[0].set_title('Gaussian Mixture')



ax[1].scatter(data[['nb_played_games']], data[['hours_played']], c=label_color, s=8, marker='o')

ax[1].set_xlabel('nb_played_games')

ax[1].set_ylabel('hours_played(log)')

ax[1].set_title('Gaussian Mixture with y log scale')

ax[1].set_yscale('log')



ax[2].scatter(data[['nb_played_games']], data[['hours_played']], c=label_color, s=8, marker='o')

ax[2].set_xlabel('nb_played_games(log)')

ax[2].set_ylabel('hours_played(log)')

ax[2].set_title('Gaussian Mixture with x and y log scale')

ax[2].set_yscale('log')

ax[2].set_xscale('log')
