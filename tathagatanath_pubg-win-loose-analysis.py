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
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
train=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
train.shape
train_copy=train.copy()
train_copy.shape
train_copy=train_copy[train_copy['winPlacePerc'].isnull()==False]
train_copy.shape
train_copy=train_copy.astype({'matchType':'category'})
train_copy.info()
index=train_copy[train_copy['matchType'].isin(['solo-fpp','solo','normal-solo-fpp','normal-solo'])].index.tolist()

for i in index :
    train_copy.at[i,'assists']=0
    train_copy.at[i,'revives']=0
    train_copy.at[i,'teamKills']=0
index=train_copy[train_copy['matchType'].isin(['duo-fpp','duo'])].index.tolist()

for i in index :
    if train_copy.at[i,'teamKills'] > 1 :
        train_copy.at[i,'teamKills']=1
train_copy.describe()
print(train_copy['assists'].value_counts())
print(train_copy['assists'].skew())
sns.boxplot(train_copy['assists'])
print(train_copy['boosts'].value_counts())
print(train_copy['boosts'].skew())
sns.boxplot(train_copy['boosts'])
train_copy=train_copy[train_copy['boosts']<25]
print(train_copy['damageDealt'].value_counts())
print(train_copy['damageDealt'].skew())
sns.boxplot(train_copy['damageDealt'])
train_copy=train_copy[train_copy['damageDealt']<6000]
print(train_copy['DBNOs'].value_counts())
print(train_copy['DBNOs'].skew())
sns.boxplot(train_copy['DBNOs'])
train_copy=train_copy[train_copy['DBNOs']<50]
print(train_copy['headshotKills'].value_counts())
print(train_copy['headshotKills'].skew())
sns.boxplot(train_copy['headshotKills'])
train_copy=train_copy[train_copy['headshotKills']<60]
print(train_copy['heals'].value_counts())
print(train_copy['heals'].skew())
sns.boxplot(train_copy['heals'])
train_copy=train_copy[train_copy['heals']<70]
print(train_copy['killPlace'].value_counts())
print(train_copy['killPlace'].skew())
sns.boxplot(train_copy['killPlace'])
m=train['killPlace'].mean()
sd=train['killPlace'].std()

train[(train['killPlace']<(m-3*sd)) | (train['killPlace']>(m+3*sd))]
print(train_copy['killPoints'].value_counts())
print(train_copy['killPoints'].skew())
sns.boxplot(train_copy['killPoints'])
m=train['killPoints'].mean()
sd=train['killPoints'].std()

train[(train['killPoints']<(m-3*sd)) | (train['killPoints']>(m+3*sd))]
print(train_copy['kills'].value_counts())
print(train_copy['kills'].skew())
sns.boxplot(train_copy['kills'])
train_copy=train_copy[train_copy['kills']<50]
print(train_copy['killStreaks'].value_counts())
print(train_copy['killStreaks'].skew())
sns.boxplot(train_copy['killStreaks'])
print(train_copy['longestKill'].value_counts())
print(train_copy['longestKill'].skew())
sns.boxplot(train_copy['longestKill'])
print(train_copy['matchDuration'].value_counts())
print(train_copy['matchDuration'].skew())
sns.boxplot(train_copy['matchDuration'])
print(train_copy['maxPlace'].value_counts())
print(train_copy['maxPlace'].skew())
sns.boxplot(train_copy['maxPlace'])
print(train_copy['numGroups'].value_counts())
print(train_copy['numGroups'].skew())
sns.boxplot(train_copy['numGroups'])
print(train_copy['rankPoints'].value_counts())
print(train_copy['rankPoints'].skew())
sns.boxplot(train_copy['rankPoints'])
print(train_copy['revives'].value_counts())
print(train_copy['revives'].skew())
sns.boxplot(train_copy['revives'])
train_copy=train_copy[train_copy['revives']<20]
print(train_copy['rideDistance'].value_counts())
print(train_copy['rideDistance'].skew())
sns.boxplot(train_copy['rideDistance'])
train_copy=train_copy[train_copy['rideDistance']<40000]
print(train_copy['roadKills'].value_counts())
print(train_copy['roadKills'].skew())
sns.boxplot(train_copy['roadKills'])
train_copy=train_copy[train_copy['roadKills']<12.5]
print(train_copy['swimDistance'].value_counts())
print(train_copy['swimDistance'].skew())
sns.boxplot(train_copy['swimDistance'])
train_copy=train_copy[train_copy['swimDistance']<3000]
print(train_copy['teamKills'].value_counts())
print(train_copy['teamKills'].skew())
sns.boxplot(train_copy['teamKills'])
print(train_copy['vehicleDestroys'].value_counts())
print(train_copy['vehicleDestroys'].skew())
sns.boxplot(train_copy['vehicleDestroys'])
print(train_copy['walkDistance'].value_counts())
print(train_copy['walkDistance'].skew())
sns.boxplot(train_copy['walkDistance'])
train_copy=train_copy[train_copy['walkDistance']<25000]
print(train_copy['weaponsAcquired'].value_counts())
print(train_copy['weaponsAcquired'].skew())
sns.boxplot(train_copy['weaponsAcquired'])
train_copy=train_copy[train_copy['weaponsAcquired']<200]
print(train_copy['winPoints'].value_counts())
print(train_copy['winPoints'].skew())
sns.boxplot(train_copy['winPoints'])
m=train['winPoints'].mean()
sd=train['winPoints'].std()

train[(train['winPoints']<(m-3*sd)) | (train['winPoints']>(m+3*sd))]
print(train_copy['winPlacePerc'].value_counts())
print(train_copy['winPlacePerc'].skew())
sns.boxplot(train_copy['winPlacePerc'])
m=train['winPlacePerc'].mean()
sd=train['winPlacePerc'].std()

train[(train['winPlacePerc']<(m-3*sd)) | (train['winPlacePerc']>(m+3*sd))]
x=train_copy.pivot_table(index=['matchType'], aggfunc='sum')
x.drop(columns=['maxPlace','numGroups','killPlace','killPoints','longestKill','matchDuration','rankPoints','winPoints','winPlacePerc'], inplace=True)
x
x=x.reset_index()
x
plt.figure(figsize=(25,6))

t=x.sort_values('assists')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='assists',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('Assists', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.4, i.get_height(), int(i.get_height()), fontsize=20, rotation=45, color='black', ha='center')
plt.figure(figsize=(25,6))

t=x.sort_values('DBNOs')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='DBNOs',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('DBNOs', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.4, i.get_height(), int(i.get_height()), fontsize=20, rotation=45, color='black', ha='center')
plt.figure(figsize=(25,6))

t=x.sort_values('boosts')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='boosts',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('Boosts', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.4, i.get_height(), int(i.get_height()), fontsize=20, rotation=45, color='black', ha='center')
plt.figure(figsize=(25,6))

t=x.sort_values('damageDealt')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='damageDealt',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('Damage Dealt', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.5, i.get_height(), int(i.get_height()), rotation=45, fontsize=20, color='black', ha='center')
plt.figure(figsize=(25,6))

t=x.sort_values('headshotKills')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='headshotKills',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('Headshot Kills', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.4, i.get_height(), int(i.get_height()), fontsize=20, rotation=45, color='black', ha='center')
plt.figure(figsize=(25,6))

t=x.sort_values('heals')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='heals',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('Heals', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.4, i.get_height(), int(i.get_height()), fontsize=20, rotation=45, color='black', ha='center')
plt.figure(figsize=(25,6))

t=x.sort_values('revives')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='revives',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('Revives', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.4, i.get_height(), int(i.get_height()), fontsize=20, rotation=45, color='black', ha='center')
plt.figure(figsize=(25,6))

t=x.sort_values('roadKills')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='roadKills',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('Road Kills', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.4, i.get_height(), int(i.get_height()), fontsize=20,rotation=45, color='black', ha='center')
plt.figure(figsize=(25,6))

t=x.sort_values('teamKills')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='teamKills',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('Team Kills', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.4, i.get_height(), int(i.get_height()), fontsize=20, rotation=45, color='black', ha='center')
plt.figure(figsize=(25,6))

t=x.sort_values('weaponsAcquired')
order=t['matchType'].values.tolist()

chart = sns.barplot(
    data=t,
    x='matchType',
    y='weaponsAcquired',
    order=order,
)

chart.set_xticklabels(t['matchType'], rotation=90, fontsize=20)
chart.set_yscale('symlog')
chart.set_xlabel('Match Type', fontsize=28)
chart.set_ylabel('Weapons Acquired', fontsize=28)

for i in chart.patches :
    chart.text(i.get_x()+.4, i.get_height(), int(i.get_height()), fontsize=20, rotation=45, color='black', ha='center')
plt.figure(figsize=(25,10))
sns.heatmap(x.corr(), cmap='autumn', annot=True, fmt='.4g')
