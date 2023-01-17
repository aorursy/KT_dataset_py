import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgbm

from sklearn.preprocessing import Normalizer



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
a = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

b = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
a.shape
b.shape
a.drop(2744604, inplace=True)
ticks = a['kills'].values

f,ax = plt.subplots(figsize = (10,10))

ax.set_xticklabels(ticks,rotation = 60)

sns.boxplot(x = 'kills',y = 'damageDealt',data = a)
f,ax = plt.subplots(figsize = (10,10))

sns.scatterplot(x = 'kills',y = 'killPlace',data = a)
no_matches = a.loc[:,'matchId'].nunique()

print(no_matches)
match_types = a['matchType'].value_counts().reset_index()

match_types.columns = ['Type','Count']

print(match_types)
ticks = match_types.Type.values

f,ax = plt.subplots(figsize = (10,10))

ax.set_xticklabels(ticks,rotation = 60)

sns.barplot(x = 'Type',y = 'Count',data = match_types)

plt.show()
match_types2 = a['matchType'].value_counts().to_frame()
agg_squads = match_types2.loc[['squad-fpp','squad','normal-squad-fpp','normal-squad'],'matchType'].sum()
agg_duos = match_types2.loc[['duo-fpp','duo','normal-duo-fpp','normal-duo'],'matchType'].sum()
agg_solos = match_types2.loc[['solo-fpp','solo','normal-solo-fpp','normal-solo'],'matchType'].sum()
agg_df = pd.DataFrame([agg_squads,agg_duos,agg_solos],index = ['Squad','duo','solo'],columns = ['Count'])

print(agg_df)
sns.distplot(a['numGroups'])

plt.show()
f,ax = plt.subplots(figsize = (10,10))

sns.boxplot(x = 'DBNOs',y = 'kills',data = a)
a[a['kills']>60][['Id','headshotKills','damageDealt','kills','killStreaks','longestKill','assists','matchType']]
f,ax = plt.subplots(figsize = (10,10))

sns.countplot(a['headshotKills'])
f,ax = plt.subplots(figsize = (10,10))

sns.countplot(a['DBNOs'])
sns.distplot(a['longestKill'])
print(a['longestKill'].mean())

print(a['longestKill'].quantile(0.95))

print(a['longestKill'].max())
n = a['walkDistance']==0

print(n.sum())

k = a['rideDistance'] == 0

print(k.sum())

x = a['swimDistance'] == 0

print(x.sum())
suspects = a.query('winPlacePerc == 1 & walkDistance == 0')

suspects.head()
print(suspects['rideDistance'].max(),suspects['swimDistance'].max())
print(a['weaponsAcquired'].mean())

print(a['weaponsAcquired'].min())

print(a['weaponsAcquired'].max())

print(a['weaponsAcquired'].quantile(0.99))
plt.hist(a['weaponsAcquired'],range = (0,10),rwidth= 0.9)

plt.show()
ax = sns.clustermap(a.corr(),annot = True,linewidths = 0.6,fmt = '.2f',figsize = (15,15))

plt.show()
top10 = a[a['winPlacePerc']>0.9]

print(top10['kills'].mean())

print(top10['kills'].min())

print(top10['kills'].max())

print(top10['kills'].quantile(0.95))
print(top10['longestKill'].mean())

print(top10['longestKill'].max())
sns.clustermap(top10.corr(),annot = True,fmt = '.2f',figsize = (15,10))

plt.show()
a.dropna(subset=["winPlacePerc"], inplace=True)
x = a.drop(['Id','matchType','matchId','groupId','winPlacePerc'],axis = 1)
x = x.iloc[:100000]
x.shape
y = a['winPlacePerc']
y = y.iloc[:100000]
y.shape
model = lgbm.LGBMRegressor(max_depth = 30,learning_rate = 0.01,n_estimators = 250)

model.fit(x,y)
lgbm.plot_importance(model)
b = b.drop(['Id','matchType','matchId','groupId'],axis = 1)
y_pred = model.predict(b)
print(y_pred)
plt.hist(y_pred,bins = 20,rwidth = 0.9)

plt.show()