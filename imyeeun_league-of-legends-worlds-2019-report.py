import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.set_option('display.max_column',None)

pd.set_option('display.max_row',None)



from matplotlib import rcParams



# figure size in inches

rcParams['figure.figsize'] = 11.7,8.27
# Import data

df = pd.read_excel('../input/league-of-legends-world-championship-2019/2019-summer-match-data-OraclesElixir-2019-11-10.xlsx')

df.head()
df.info()
df.shape
teamdf = df.loc[df['position']=='Team',:]

teamdf.head()
teamdf.shape
playersdf = df.loc[df['position']!='Team',:]

playersdf.head()
playersdf.shape
bans3 = teamdf[['ban1', 'ban2', 'ban3']].melt()

bans3.head()
bans3['value'].value_counts()[:10].plot(kind='barh', figsize=(8, 6))
playersdf['champion'].value_counts()[:10].plot(kind='barh', figsize=(8, 6))
playersdf.groupby(['champion','position'])['position'].count()
pd.crosstab(playersdf.champion,playersdf.position).T.style.background_gradient(cmap='summer_r')
teamdf.groupby(['team','result'])['result'].count()
pd.crosstab(teamdf.team,teamdf.result).T.style.background_gradient(cmap='summer_r')
lckdf = df.loc[df['team'].isin(['Damwon Gaming', 'Griffin','SK Telecom T1']) ,:]

lckdf.head()
lck_team_df = lckdf[lckdf['player']=='Team']
lck_players_df = lckdf[lckdf['player']!='Team']
pd.crosstab([lck_team_df.team,lck_team_df.result],lck_team_df.side,margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('side','result',hue = 'team',data = lck_team_df)

plt.show()
lck_players_df['kda'] = (lck_players_df['k'] + lck_players_df['a'])/lck_players_df['d']
lck_players_df['kda']=lck_players_df['kda'].replace(np.inf,(lck_players_df['k'] + lck_players_df['a']))
lck_players_df
lck_players_df.groupby('player')['kda'].mean().nlargest(5)
lck_players_df.groupby(['player','champion'])['kda'].max().nlargest(5)