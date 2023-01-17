import numpy as np

import pandas as pd

import re

import seaborn as sns

from matplotlib import pyplot as plt

df = pd.read_csv("../input/steam-200k.csv", header=None, index_col=None, names=['UserID', 'Game', 'Action', 'Hours', 'Other'])

df.head()
df.describe()
df.Action.unique()
len(df.Game.unique())
len(df.UserID.unique())
actions = df.groupby('Action')['Action'].agg('count')

actions = pd.DataFrame({'action': actions.index, 'n': actions.values})

sns.barplot(x = 'action', y = 'n', data = actions)
game_counts = df.groupby('Game')['Game'].agg('count').sort_values(ascending=False)

game_counts = pd.DataFrame({'game': game_counts.index, 'n': game_counts.values})[0:20]

sns.barplot(y = 'game', x = 'n', data = game_counts)
game_counts = df.groupby('Game')['Hours'].agg(np.sum).sort_values(ascending=False)

game_counts = pd.DataFrame({'game': game_counts.index, 'hours': game_counts.values})[0:20]

sns.barplot(y = 'game', x = 'hours', data = game_counts)
user_counts = df.groupby('UserID')['Hours'].agg(np.sum).sort_values(ascending=False)

user_counts = pd.DataFrame({'user': user_counts.index, 'hours': user_counts.values})[0:50]

user_counts['hours'].plot.bar()

plt.title('Top 50 users hours played')

plt.ylabel('Hours')
user_counts = df.groupby('UserID')['UserID'].agg('count').sort_values(ascending=False)

user_df = pd.DataFrame({'user': user_counts.index, 'n': user_counts.values})

user_df.head()
n_events = np.sum(user_df['n'])

csum = np.cumsum(100.0*user_df['n']/n_events)



# 80% of data

threshold = 80.0



threshold_idx = np.argmin(np.abs(csum-threshold))

plt.plot(csum)

plt.axvline(threshold_idx, color='r')

plt.title('cumulative sum of user event counts')

print('We have found that {} users out of {} ({}%) in total explain {}% of the events in the data.'.format(threshold_idx, user_df.shape[0], 100.0*threshold_idx/user_df.shape[0], threshold))
game_counts = df.groupby('Game')['Game'].agg('count').sort_values(ascending=False)

game_df = pd.DataFrame({'game': game_counts.index, 'n': game_counts.values})

game_df.head()
n_events = np.sum(game_df['n'])

csum = np.cumsum(100.0*game_df['n']/n_events)



# 80% of data

threshold = 80.0



threshold_idx = np.argmin(np.abs(csum-threshold))

plt.plot(csum)

plt.axvline(threshold_idx, color='r')

plt.title('cumulative sum of game event counts')

print('We have found that {} games out of {} ({}%) in total explain {}% of the events in the data.'.format(threshold_idx, game_df.shape[0], 100.0*threshold_idx/user_df.shape[0], threshold))
top_users = df.groupby('UserID')['Hours'].agg(np.sum).sort_values(ascending=False).index[0:100]

top_games = df.groupby('Game')['Hours'].agg(np.sum).sort_values(ascending=False).index[0:50]
dff = df[df['UserID'].isin(top_users)]

dff = dff[dff['Game'].isin(top_games)]

pivoted = pd.pivot_table(dff, values='Hours', index=['UserID'], columns=['Game'], aggfunc=np.sum).fillna(0)

s = pivoted.max().max()

pivoted = pivoted / s

pivoted.shape
sns.clustermap(pivoted, figsize=(12,8))