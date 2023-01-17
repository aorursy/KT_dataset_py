

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/fifa-21-complete-player-dataset/fifa21_male2.csv")

cols=['Name', 'Age', 'OVA', 'Nationality', 'foot','Value','BP','Height','PAC','SHO','PAS','DRI','DEF','PHY']

df=df[cols]

df.head()
df.shape

df.info
df.describe
df.dtypes

df['Age'].describe()
df['Age'].plot(kind='box', vert=False, figsize=(14,6))


df['Age'].plot(kind='density', figsize=(14,6))
df['Nationality'].value_counts()

df['BP'].value_counts().plot(kind='pie', figsize=(6,6))
ax = df['BP'].value_counts().plot(kind='bar', figsize=(14,6))

ax.set_ylabel('Value')

ax.set_xlabel(' Main Position')
df[df['BP'] == 'CB'] # keep only Center Back


corr = df[df['BP'] == 'CB'].corr() #Correlation matrix for CB player

corr
fig = plt.figure(figsize=(8,8))

plt.matshow(corr, cmap='RdBu', fignum=fig.number)

plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');

plt.yticks(range(len(corr.columns)), corr.columns);


corr = df[df['BP'] == 'ST'].corr() #Correlation matrix for Striker player

corr
fig = plt.figure(figsize=(8,8))

plt.matshow(corr, cmap='RdBu', fignum=fig.number)

plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');

plt.yticks(range(len(corr.columns)), corr.columns);


corr = df[df['BP'] == 'GK'].corr() #Correlation matrix for Goal Keeper player

corr
fig = plt.figure(figsize=(8,8))

plt.matshow(corr, cmap='RdBu', fignum=fig.number)

plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');

plt.yticks(range(len(corr.columns)), corr.columns);
corr = df[df['BP'] == 'RW'].corr() #Correlation matrix for Goal Keeper player

corr
fig = plt.figure(figsize=(8,8))

plt.matshow(corr, cmap='RdBu', fignum=fig.number)

plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');

plt.yticks(range(len(corr.columns)), corr.columns);
df.loc[(df['foot'] == 'Right'), 'OVA'].mean() # right foot average player
df.loc[(df['foot'] == 'Left'), 'OVA'].mean() # left foot average player 
ax = df[['OVA', 'Age']].boxplot(by='Age', figsize=(10,6))

ax.set_ylabel('OVA')

df.plot(kind='scatter', x='Age', y='OVA', figsize=(6,6))


ax = df[['OVA', 'BP']].boxplot(by='BP', figsize=(10,6)) # niveau moyen des joueurs en fonction des positions

ax.set_ylabel('OVA')



dt =df[df.OVA > 84]

dt.shape



ax = dt[['OVA', 'BP']].boxplot(by='BP', figsize=(10,6)) # niveau moyen des joueurs en fonction des positions

ax.set_ylabel('OVA') # repartition of the role of super player ( OVA > 84)
