import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=Warning)
train = pd.read_csv('../input/train.csv')
display(train.head())
display(train.describe())
def show_countplot(column):
    plt.figure(figsize=(12,4))
    sns.countplot(data=train, x=column).set_title(column)
    plt.show()
    
def show_distplot(column):
    plt.figure(figsize=(12, 4))
    sns.distplot(train[column], bins=50)
    plt.show()
show_countplot('kills')
train[train['kills'] >= 40]
train['headshot_rate'] = train['headshotKills'] / train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)
show_distplot('headshot_rate')
train[(train['headshot_rate'] >= 1) & (train['kills'] >= 10)]
show_distplot('longestKill')
train[train['longestKill'] >= 1000]
show_countplot('teamKills')
train[train['teamKills'] >= 5]
train[['walkDistance', 'rideDistance', 'swimDistance']].describe()
show_distplot('walkDistance')
train[train['walkDistance'] >= 13000]
show_distplot('rideDistance')
train[train['rideDistance'] >= 30000]
show_countplot('weaponsAcquired')
train[train['weaponsAcquired'] >= 60]
show_countplot('heals')
train[train['heals'] >= 50]
agg = train.groupby(['groupId']).size().to_frame('players_in_team')
train = train.merge(agg, how='left', on=['groupId'])
train[['matchId', 'groupId', 'players_in_team']].head()
agg = train.groupby(['matchId']).agg({'players_in_team': ['min', 'max', 'mean']})
agg.columns = ['_'.join(x) for x in agg.columns.ravel()]
train = train.merge(agg, how='left', on=['matchId'])
train['players_in_team_var'] = train.groupby(['matchId'])['players_in_team'].var()
display(train[['matchId', 'players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']].head())
display(train[['matchId', 'players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']].describe())
plt.figure(figsize=(10,4))
for i, match_id in enumerate(train.nlargest(2, 'players_in_team_var')['matchId'].values):
    plt.subplot(1, 2, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team'])
plt.show()
plt.figure(figsize=(20,5))
for i, match_id in enumerate(train[train['players_in_team_max'] == 1]['matchId'].values[:4]):
    plt.subplot(1, 4, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team'])
plt.show()
plt.figure(figsize=(20,5))
for i, match_id in enumerate(train[(train['players_in_team_max'] == 2) & (train['players_in_team_var'] == 0)]['matchId'].values[:4]):
    plt.subplot(1, 4, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team'])
plt.show()
plt.figure(figsize=(20,5))
for i, match_id in enumerate(train[(train['players_in_team_max'] == 4) & (train['players_in_team_var'] > 0)]['matchId'].values[:4]):
    plt.subplot(1, 4, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team'])
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
train['players_in_team_var'] = train['players_in_team_var'].fillna(-1)
columns = ['players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']
data = train.groupby(['matchId']).first()[columns].reset_index()
preprocessor = make_pipeline(StandardScaler(), PCA(n_components=2))
reduced_data = preprocessor.fit_transform(data)
model = KMeans(n_clusters=8)
model.fit(reduced_data)
data['game_mode'] = model.predict(reduced_data)

plt.figure(figsize=(6,6))
plt.scatter(reduced_data[:,0], reduced_data[:,1], c=data['game_mode'])
plt.show()
data.groupby(['game_mode']).mean().reset_index()[
    ['game_mode', 'players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']
].merge(data.groupby(['game_mode']).size().to_frame('count'), how='left', on=['game_mode'])
plt.figure(figsize=(20,10))
for i, match_id in enumerate(data.groupby(['game_mode']).first()['matchId'].values):
    plt.subplot(2, 4, i + 1)
    sns.distplot(train[train['matchId'] == match_id]['players_in_team']).set_title(f'game_mode: {i}')
plt.show()