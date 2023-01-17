# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.figure_factory as ff





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
pubg=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
pubg.head()
pubg.info()
pubg.isnull().sum()
for i in pubg.columns[3:]:

    print("min of ",i,min(pubg[i]))

    print("max of ",i,max(pubg[i]))
pubg['rankPoints'].value_counts()
pubg.info()
pubg['matchType'].unique()
pubg.drop(columns={'rankPoints'},inplace=True)

pubg=pubg[~pubg['winPlacePerc'].isnull()]
pubg.isnull().sum()
pubg['matchType']=pubg['matchType'].astype('category')

pubg.info()
pubg['playersJoined'] = pubg.groupby('matchId')['matchId'].transform('count')

pubg
pubg=pubg[pubg['playersJoined']>97]

pubg.shape
pubg['totalDistance'] = pubg['rideDistance'] + pubg['walkDistance'] + pubg['swimDistance']
pubg[(pubg['kills'] > 0) & (pubg['totalDistance'] == 0)].shape[0]
pubg=pubg[~((pubg['kills'] > 0) & (pubg['totalDistance'] == 0))]

pubg.shape
pubg_win=pubg[pubg['winPlacePerc']>0.96]

pubg_win.shape
pubg_lose=pubg[pubg['winPlacePerc']<0.4]

pubg_lose.shape
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['assists'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['assists'])

plt.show()
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['boosts'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['boosts'])

plt.show()
plt.subplots(figsize=(20,10))

sns.distplot(pubg_win['damageDealt'],label="Losing Players")

sns.distplot(pubg_lose['damageDealt'],label="Losing Players")

plt.legend(["Players with win chance of 97% and above","Players with win chance of 3% and below"])

plt.show()
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['DBNOs'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['DBNOs'])

plt.show()
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['headshotKills'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['headshotKills'])

plt.show()
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['heals'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['heals'])
plt.subplots(figsize=(20,10))

sns.distplot(pubg_win['killPlace'])

sns.distplot(pubg_lose['killPlace'])

plt.legend(["Players with win chance of 97% and above","Players with win chance of 3% and below"])

plt.show()
plt.subplots(figsize=(20,10))

sns.distplot(pubg_win['kills'])

sns.distplot(pubg_lose['kills'])

plt.legend(["Players with win chance of 97% and above","Players with win chance of 3% and below"])

plt.show()
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['killStreaks'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['killStreaks'])
plt.subplots(figsize=(20,10))

sns.distplot(pubg_win['longestKill'])

sns.distplot(pubg_lose['longestKill'])

plt.legend(["Players with win chance of 97% and above","Players with win chance of 3% and below"])

plt.show()
plt.subplots(figsize=(40,30))

plt.subplot(2, 1, 1)

plt.title("Players with win chance of 97% and above")

sns.countplot(x=pubg_win['matchType'],order=pubg_win['matchType'].value_counts().index)

print("")

plt.subplot(2, 1, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(x=pubg_lose['matchType'],order=pubg_lose['matchType'].value_counts().index)

plt.show()
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['revives'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['revives'])

plt.show()
plt.subplots(figsize=(20,10))



sns.distplot(pubg_win['totalDistance'])

plt.title("Players with win chance of 99% and above")



plt.title("Players with win chance of 1% and below")

sns.distplot(pubg_lose['totalDistance'])

plt.show()
plt.subplots(figsize=(30,20))

sns.distplot(pubg_win['walkDistance'],label="walkDistance_win", kde=False)

sns.distplot(pubg_win['rideDistance'],label="rideDistance_lose", kde=False)

sns.distplot(pubg_win['swimDistance'],label="swimDistance_win", kde=False)

plt.legend()

plt.show()
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['roadKills'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['roadKills'])

plt.show()
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['vehicleDestroys'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['vehicleDestroys'])

plt.show()
plt.subplots(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(pubg_win['teamKills'])

plt.title("Players with win chance of 97% and above")

plt.subplot(1, 2, 2)

plt.title("Players with win chance of 3% and below")

sns.countplot(pubg_lose['teamKills'])

plt.show()
plt.subplots(figsize=(20, 20))

sns.heatmap(pubg.corr(), annot=True, linewidths=.5, fmt= '.1f')

plt.show()
plt.subplots(figsize=(15, 15))

cols=pubg.corr().nlargest(6, 'winPlacePerc')['winPlacePerc'].index

cm = np.corrcoef(pubg[cols].values.T)

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.relplot(x='walkDistance',y='winPlacePerc',kind='scatter',data=pubg,height=8)

plt.grid()

plt.show()
sns.relplot(x='totalDistance',y='winPlacePerc',kind='scatter',data=pubg,height=8)

plt.grid()

plt.show()
plt.subplots(figsize=(15, 15))

sns.pointplot(x='boosts',y='winPlacePerc',data=pubg,height=8)

plt.grid()

plt.show()


plt.subplots(figsize=(15, 15))

sns.pointplot(x='weaponsAcquired',y='winPlacePerc',data=pubg,height=8)

plt.grid()

plt.show()
plt.subplots(figsize=(15, 15))

sns.pointplot(x='kills',y='winPlacePerc',data=pubg,height=8)

plt.grid()

plt.show()
plt.subplots(figsize=(15, 15))

sns.pointplot(x='DBNOs',y='winPlacePerc',data=pubg,height=8)

plt.grid()

plt.show()
plt.subplots(figsize=(15, 15))

sns.pointplot(x='assists',y='winPlacePerc',data=pubg,height=8)

plt.grid()

plt.show()
plt.subplots(figsize=(15, 15))

sns.pointplot(x='heals',y='winPlacePerc',data=pubg,height=8)

plt.grid()

plt.show()
plt.subplots(figsize=(15, 15))

sns.pointplot(x='revives',y='winPlacePerc',data=pubg,height=8)

plt.grid()

plt.show()