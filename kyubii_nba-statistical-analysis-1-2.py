import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import plotting
from scipy import stats
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from scipy import stats

sns.set(style="darkgrid", color_codes = True)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
players = pd.read_csv('../input/nba-players-stats/Players.csv')
players.drop(columns = ['Unnamed: 0'], axis = 'column', inplace = True)
players.head()
players.describe()
players_data = pd.read_csv('../input/nba-players-stats/player_data.csv')
players_data.rename(columns={'name': 'Player'}, inplace=True)
players_data.fillna('No College')
players_data.head()
players_data.describe()
final_df = pd.read_csv('../input/nba-players-stats/Seasons_Stats.csv')
final_df.drop(columns = ['Unnamed: 0'], axis = 'column', inplace = True)
final_df.head()
final_df.columns
final_df.describe()
position = players_data.groupby('position')['Player'].count().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x=position.index, y=position.values)

plt.title("Position Distribution")
plt.show()
fig, ax = plt.subplots(figsize=(14, 10))
sns.distplot(final_df['Age']);

plt.title("Age Distribution")
plt.show()
import random
def random_colors(number_of_colors=1):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color
final_df['ppg'] = final_df.PTS/final_df.G
ppg = pd.DataFrame()
ppg['player'] = final_df.groupby('Player').mean()['ppg'].index
ppg['ppg'] = final_df.groupby('Player').mean()['ppg'].values

ppg = ppg.sort_values('ppg', ascending=False).head(10)
ppg
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x="ppg", y="player", hue="player", data=ppg, dodge=False)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Top 10 Players with Highest PPG")
plt.show()
fig, ax = plt.subplots(figsize=(14, 10))  
for key, val in ppg[['player']].iterrows():
    sns.lineplot(x="Year", y="ppg", data=final_df[final_df['Player'] == val['player']])
    
plt.legend(ppg['player'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Highest PPG Trends")
starter = final_df[(final_df['GS'] > 50) & (final_df['2PA'] > 100)]

two_P = pd.DataFrame()
two_P['player'] = starter.groupby('Player').mean()['2P%'].index
two_P['2P%'] = starter.groupby('Player').mean()['2P%'].values

two_P = two_P.sort_values('2P%', ascending=False).head(10)
two_P
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x="2P%", y="player", hue="player", data=two_P, dodge=False)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Top 10 Players with Highest Two-Point %")
plt.show()
fig, ax = plt.subplots(figsize=(14, 10))  
for key, val in two_P[['player']].iterrows():
    sns.lineplot(x="Year", y="2P%", data=final_df[final_df['Player'] == val['player']])
    
plt.legend(two_P['player'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Highest 2PT % Trends")
starter = final_df[(final_df['GS'] > 50) & (final_df['3PA'] > 100)]

three_P = pd.DataFrame()
three_P['player'] = starter.groupby('Player').mean()['3P%'].index
three_P['3P%'] = starter.groupby('Player').mean()['3P%'].values

three_P = three_P.sort_values('3P%', ascending=False).head(10)
three_P
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x="3P%", y="player", hue="player", data=three_P, dodge=False)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Top 10 Players with Highest 3-Point %")
plt.show()
fig, ax = plt.subplots(figsize=(14, 10))  
for key, val in three_P[['player']].iterrows():
    sns.lineplot(x="Year", y="2P%", data=final_df[final_df['Player'] == val['player']])
    
plt.legend(three_P['player'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Highest 3PT % Trends")
starter = final_df[(final_df['GS'] > 50) & (final_df['FTA'] > 100)]

ft_P = pd.DataFrame()
ft_P['player'] = starter.groupby('Player').mean()['FT%'].index
ft_P['FT%'] = starter.groupby('Player').mean()['FT%'].values

ft_P = ft_P.sort_values('FT%', ascending=False).head(10)
ft_P
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x="FT%", y="player", hue="player", data=ft_P, dodge=False)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Top 10 Players with Highest FT %")
plt.show()
final_df['apg'] = final_df.AST/final_df.G
apg = pd.DataFrame()
apg['player'] = final_df.groupby('Player').mean()['apg'].index
apg['apg'] = final_df.groupby('Player').mean()['apg'].values

apg = apg.sort_values('apg', ascending=False).head(10)
apg
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x="apg", y="player", hue="player", data=apg, dodge=False)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Top 10 Players with Highest APG")
plt.show()
fig, ax = plt.subplots(figsize=(14, 10))  
for key, val in apg[['player']].iterrows():
    sns.lineplot(x="Year", y="apg", data=final_df[final_df['Player'] == val['player']])
    
plt.legend(apg['player'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Highest APG Trends")
final_df['rpg'] = final_df.TRB/final_df.G
rpg = pd.DataFrame()
rpg['player'] = final_df.groupby('Player').mean()['rpg'].index
rpg['rpg'] = final_df.groupby('Player').mean()['rpg'].values

rpg = rpg.sort_values('rpg', ascending=False).head(10)
rpg
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x="rpg", y="player", hue="player", data=rpg, dodge=False)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Top 10 Players with Highest RPG")
plt.show()
fig, ax = plt.subplots(figsize=(14, 10))  
for key, val in rpg[['player']].iterrows():
    sns.lineplot(x="Year", y="rpg", data=final_df[final_df['Player'] == val['player']])
    
plt.legend(rpg['player'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title("Highest RPG Trends")
final_df['tpg'] = final_df.TOV/final_df.G
tpg = pd.DataFrame()
tpg['player'] = final_df.groupby('Player').mean()['tpg'].index
tpg['tpg'] = final_df.groupby('Player').mean()['tpg'].values

tpg = tpg.sort_values('tpg', ascending=False).head(10)
tpg

fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x="tpg", y="player", hue="player", data=tpg, dodge=False)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Top 10 Players with Highest TPG")
plt.show()
