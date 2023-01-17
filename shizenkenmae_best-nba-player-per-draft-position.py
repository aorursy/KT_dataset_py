import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import os

%matplotlib inline
df = pd.read_csv("../input/nba-players-data/all_seasons.csv")
df.info()
df.describe()
test_feat = ['gp','net_rating']

df.loc[df.player_name=='Bruce Bowen',test_feat]
(df.loc[(df['gp']==1 )& (df['net_rating']>20)])
df.draft_year.unique()
sns.barplot(y=df.loc[df.season=='2018-19'].draft_year.value_counts().index,x=df.loc[df.season=='2018-19'].draft_year.value_counts())
total_player = len(df.loc[df.season=='2018-19'].player_name.unique())

undrafted_player = len(df.loc[(df.season=='2018-19')&(df.draft_year=='Undrafted')].player_name.unique())

prcntg  = 100*(undrafted_player/total_player)

print(prcntg)
df.drop(df[df.draft_year<'1995'].index, inplace=True)
df['draft_number'].replace('Undrafted','82',inplace=True)

df['draft_number'].replace('82','61',inplace=True)

df['draft_number'] = pd.to_numeric(df['draft_number'])
df_player = df[['player_name','gp']].groupby('player_name').sum().reset_index()

df_player = df_player.loc[df_player['gp'] < 5]

for p in df_player['player_name']:

    df.drop(df[df.player_name==p].index, inplace=True)
stats = ['gp', 'pts', 'reb', 'ast', 'net_rating',

       'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']

avg_per_pick = df.groupby(['draft_number'])[stats].mean().reset_index()

avg_per_player = df.groupby(['player_name','draft_number'])[stats].mean().reset_index()
sns.regplot(x='draft_number',y='pts',data=avg_per_pick,order=3)

plt.show()
sns.regplot(x='draft_number',y='ast',data=avg_per_pick,order=3)

plt.show()
sns.regplot(x='draft_number',y='reb',data=avg_per_pick,order=3)

plt.show()
ax = sns.lineplot(x='draft_number',y='pts',data=avg_per_pick, label='pts')

ax = sns.lineplot(x='draft_number',y='ast',data=avg_per_pick, label='ast')

ax = sns.lineplot(x='draft_number',y='reb',data=avg_per_pick, label='reb')

ax.set(ylabel = 'avg')

ax.legend()

plt.show()
avg_per_pick[['draft_number','pts','reb','ast']][:15]
avg_per_pick[['draft_number','pts','reb','ast']][-5:]
avg_per_player[avg_per_player.draft_number==57]
avg_per_player[avg_per_player.draft_number==60]
ax = sns.lineplot(x='season',y='pts',data=df.loc[df['player_name'] == 'Isaiah Thomas'], label='pts')

ax = sns.lineplot(x='season',y='ast',data=df.loc[df['player_name'] == 'Isaiah Thomas'], label='ast')

ax = sns.lineplot(x='season',y='reb',data=df.loc[df['player_name'] == 'Isaiah Thomas'], label='reb')

ax.set(ylabel = 'avg')

ax.legend()

plt.show()
ax = sns.lineplot(x='season',y='pts',data=df.loc[df['player_name'] == 'Manu Ginobili'], label='pts')

ax = sns.lineplot(x='season',y='ast',data=df.loc[df['player_name'] == 'Manu Ginobili'], label='ast')

ax = sns.lineplot(x='season',y='reb',data=df.loc[df['player_name'] == 'Manu Ginobili'], label='reb')

ax.set(ylabel = 'avg')

ax.legend()

plt.show()
def score(a, b):

    #function to calculate score

    sum = 0.0

    for i in range(1,len(a)):

        sum += (b[i+1]-a[i])

    return (sum)
dist = []

for p in range(avg_per_player.shape[0]):

    val = score(avg_per_pick.loc[avg_per_player.loc[p][1]-1],avg_per_player.loc[p])

    dist.append(val)

avg_per_player['score'] = dist
avg_per_player.loc[avg_per_player.draft_number==1].sort_values('score')
avg_per_player.loc[avg_per_player.groupby('draft_number')['score'].idxmax()]
avg_per_player.loc[avg_per_player.draft_number==8].sort_values('score')
avg_per_player.loc[avg_per_player.draft_number==13].sort_values('score')
avg_per_player.loc[avg_per_player.groupby('draft_number')['score'].idxmin()][:15]
avg_per_player.loc[avg_per_player.draft_number==7].sort_values('score')