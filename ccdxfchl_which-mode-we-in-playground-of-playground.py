# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
train.tail()
train.shape
test = pd.read_csv('../input/test.csv')
test.head()
test.shape
[i for i in train.columns if i not in test.columns]
train.winPlacePerc.head()
Y_train = train['winPlacePerc']
X_train = train.drop('winPlacePerc',1)
data_all = pd.concat([X_train,test],0)
data_all.shape
data_all.isnull().sum().sort_values(ascending = False).head(2)
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
corrmat = train.corr()
k = 25
plt.subplots(figsize=(20, 20))
cols = corrmat.nlargest(k,'winPlacePerc')['winPlacePerc'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.figure(figsize=(12,6))
sns.distplot(train[['numGroups']])
# sns.distplot(train[['maxPlace']])
plt.figure(figsize=(12,6))
sns.distplot(train[['maxPlace']])
plt.axvline(50,color = 'r')
plt.axvline(25,color = 'r')
most_players_perteam = train.groupby(['matchId','groupId'],as_index = False).count()[['matchId','Id']].groupby('matchId').max()
most_players_perteam.sort_values(by='Id',ascending = False)
solo_games_matchId = most_players_perteam[most_players_perteam['Id']==1].index
duo_games_matchId = most_players_perteam[most_players_perteam['Id']==2].index
squa_games_matchId = most_players_perteam[most_players_perteam['Id']==4].index

solo_games = train[train.matchId.isin(solo_games_matchId)]
duo_games = train[train.matchId.isin(duo_games_matchId)]
squa_games = train[train.matchId.isin(squa_games_matchId)]

unknow_games = train[~(train.matchId.isin(solo_games_matchId) | (train.matchId.isin(duo_games_matchId)) | (train.matchId.isin(squa_games_matchId)))]

plt.figure(figsize=(16,15))
plt.subplot(311)
sns.distplot(train[['numGroups']])
sns.distplot(train[['maxPlace']])
plt.title('all')
plt.subplot(323)
sns.distplot(solo_games[['numGroups']])
sns.distplot(solo_games[['maxPlace']])
plt.title('Solo')
plt.subplot(324)
sns.distplot(duo_games[['numGroups']])
sns.distplot(duo_games[['maxPlace']])
plt.title('Duo')
plt.subplot(325)
sns.distplot(squa_games[['numGroups']])
sns.distplot(squa_games[['maxPlace']])
plt.title('Squa')
plt.subplot(326)
sns.distplot(unknow_games[['numGroups']])
sns.distplot(unknow_games[['maxPlace']])
plt.title('Unknow')
squa_games[(squa_games.matchId == 7)&(squa_games.groupId==2612504)]
squa_games[(squa_games.matchId == 190)&(squa_games.groupId.isin([401596,401598,401582]))].sort_values(by='groupId')
most_players_perteam = train.groupby(['matchId','groupId'],as_index = False).count()[['matchId','Id']].groupby('matchId').median()
# most_players_perteam.sort_values(by='Id',ascending = False)
solo_games_matchId = most_players_perteam[most_players_perteam['Id']==1].index
duo_games_matchId = most_players_perteam[most_players_perteam['Id']==2].index
squa_games_matchId = most_players_perteam[(most_players_perteam['Id']<=4)&(most_players_perteam['Id']>=3)].index

solo_games = train[train.matchId.isin(solo_games_matchId)]
duo_games = train[train.matchId.isin(duo_games_matchId)]
squa_games = train[train.matchId.isin(squa_games_matchId)]

unknow_games = train[~(train.matchId.isin(solo_games_matchId) | (train.matchId.isin(duo_games_matchId)) | (train.matchId.isin(squa_games_matchId)))]

plt.figure(figsize=(16,15))
plt.subplot(311)
sns.distplot(train[['numGroups']])
sns.distplot(train[['maxPlace']])
plt.title('all')
plt.subplot(323)
sns.distplot(solo_games[['numGroups']])
sns.distplot(solo_games[['maxPlace']])
plt.title('Solo')
plt.subplot(324)
sns.distplot(duo_games[['numGroups']])
sns.distplot(duo_games[['maxPlace']])
plt.title('Duo')
plt.subplot(325)
sns.distplot(squa_games[['numGroups']])
sns.distplot(squa_games[['maxPlace']])
plt.title('Squa')
plt.subplot(326)
sns.distplot(unknow_games[['numGroups']])
sns.distplot(unknow_games[['maxPlace']])
plt.title('Unknow')
most_players_perteam = train.groupby(['matchId','groupId'],as_index = False).count()[['matchId','Id']].groupby('matchId').agg(lambda x: np.mean(pd.Series.mode(x)))
# most_players_perteam.sort_values(by='Id',ascending = False)
solo_games_matchId = most_players_perteam[most_players_perteam['Id']==1].index
duo_games_matchId = most_players_perteam[most_players_perteam['Id']==2].index
squa_games_matchId = most_players_perteam[(most_players_perteam['Id']<=4)&(most_players_perteam['Id']>=3)].index

solo_games = train[train.matchId.isin(solo_games_matchId)]
duo_games = train[train.matchId.isin(duo_games_matchId)]
squa_games = train[train.matchId.isin(squa_games_matchId)]

unknow_games = train[~(train.matchId.isin(solo_games_matchId) | (train.matchId.isin(duo_games_matchId)) | (train.matchId.isin(squa_games_matchId)))]

plt.figure(figsize=(16,15))
plt.subplot(311)
sns.distplot(train[['numGroups']])
sns.distplot(train[['maxPlace']])
plt.title('all')
plt.subplot(323)
sns.distplot(solo_games[['numGroups']])
sns.distplot(solo_games[['maxPlace']])
plt.title('Solo')
plt.subplot(324)
sns.distplot(duo_games[['numGroups']])
sns.distplot(duo_games[['maxPlace']])
plt.title('Duo')
plt.subplot(325)
sns.distplot(squa_games[['numGroups']])
sns.distplot(squa_games[['maxPlace']])
plt.title('Squa')
plt.subplot(326)
sns.distplot(unknow_games[['numGroups']])
sns.distplot(unknow_games[['maxPlace']])
plt.title('Unknow')