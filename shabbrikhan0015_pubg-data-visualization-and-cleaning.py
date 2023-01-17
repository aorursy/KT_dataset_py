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
import numpy as np
import pandas as pd
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#Read Data
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.info()
train.columns
train.head()
# Check row with NaN value
train[train['winPlacePerc'].isnull()]
# Delete this player
train.drop(2744604, inplace=True)
# playersJoined
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
plt.figure(figsize=(15,10))
sns.countplot(train[train['playersJoined']>=75]['playersJoined'])
plt.title('playersJoined')
plt.show()
weapon = train[train['weaponsAcquired']<=20]
sns.countplot(x='weaponsAcquired',data =weapon)
train['healsandboosts'] = train['heals'] + train['boosts']
train[['heals', 'boosts', 'healsandboosts']].tail()
train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))


display(train[train['killsWithoutMoving'] == True].shape)
train[train['killsWithoutMoving'] == True].head(10)
# Remove 
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
train[train['roadKills'] > 10]
plt.figure(figsize=(12,4))
sns.countplot(data=train, x=train['kills']).set_title('Kills')
plt.show()
# Players who got more than 30 kills
display(train[train['kills'] > 30].shape)
train[train['kills'] > 30].head(10)
# Remove outliers
train.drop(train[train['kills'] > 30].index, inplace=True)
plt.figure(figsize=(12,4))
sns.distplot(train['longestKill'], bins=10)
plt.show()
display(train[train['longestKill'] >= 1000].shape)
train[train['longestKill'] >= 1000].head(10)
train[['walkDistance', 'rideDistance', 'swimDistance', 'totalDistance']].describe()
plt.figure(figsize=(12,4))
sns.distplot(train['walkDistance'], bins=10)
plt.show()
train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)
plt.figure(figsize=(12,4))
sns.distplot(train['rideDistance'], bins=10)
plt.show()
train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
plt.figure(figsize=(12,4))
sns.distplot(train['swimDistance'], bins=10)
plt.show()
train[train['swimDistance'] >= 2000]
# Remove outliers
train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
plt.figure(figsize=(12,4))
sns.distplot(train['weaponsAcquired'], bins=100)
plt.show()
display(train[train['weaponsAcquired'] >= 80].shape)
train[train['weaponsAcquired'] >= 80].head()
# Remove outliers
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)