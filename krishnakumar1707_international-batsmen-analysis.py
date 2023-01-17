import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
players = pd.read_csv('../input/international-cricket-players-data/personal_male.csv',parse_dates=True)
players.head()
players.info()
diff_c =players[players['nationalTeam'] != players['country']].groupby('nationalTeam',as_index=False).count().sort_values('name',ascending=False)
plt.figure(figsize=(15,15))

sns.heatmap(players.pivot_table(index='nationalTeam',columns='country',values='name',aggfunc='size'),annot=True)
plt.figure(figsize=(10,8))

sns.barplot(x='name',y='nationalTeam',data=diff_c,orient='h')
plt.figure(figsize=(20,6))

sns.countplot(players[(players['nationalTeam'] != players['country'])&(players['country']=='India')]['nationalTeam'],orient='v')
India=players[players['nationalTeam']=='India']

India['born']=pd.DatetimeIndex(India['dob']).year
plt.figure(figsize=(20,6))

sns.countplot(x='born',data=India[India['born']>1980],hue='battingStyle')
India.head()
India.groupby('battingStyle').count()['name'].plot(kind='bar')
plt.figure(figsize=(20,6))

India.groupby('bowlingStyle').count()['name'].plot(kind='bar')
plt.figure(figsize=(20,6))

g=sns.FacetGrid(India,col='battingStyle')

g.map(sns.countplot,x='born',data=India[India['born']>1980])
lefties=players.pivot_table(index='nationalTeam',columns='battingStyle',values='name',aggfunc='size')

lefties['leftie%']=lefties['Left-hand bat']/(lefties['Left-hand bat']+lefties['Right-hand bat'])*100
plt.figure(figsize=(10,8))

sns.barplot(y=lefties.index,x=lefties['leftie%'],data=lefties[lefties['Left-hand bat']>20],orient='h')
def get_len(x):

    return len(x.split(','))
players['team_count']=players['teams'].apply(get_len)
players.groupby('nationalTeam').mean()
plt.figure(figsize=(6,20))

sns.heatmap(India.pivot_table(index='bowlingStyle',columns='battingStyle',values='name',aggfunc='size'),annot=True,cmap='viridis')
plt.figure(figsize=(10,8))

sns.countplot(y='nationalTeam',data=players,orient='h')