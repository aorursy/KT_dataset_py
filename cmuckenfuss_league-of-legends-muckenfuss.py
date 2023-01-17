# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

import eli5

from eli5.sklearn import PermutationImportance

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
lol_column_path = '../input/leagueoflegends/_columns.csv'

lol_column = pd.read_csv(lol_column_path)

lol_ban_path = '../input/leagueoflegends/bans.csv'

lol_ban = pd.read_csv(lol_ban_path)

lol_gold_path = '../input/leagueoflegends/gold.csv'

lol_gold = pd.read_csv(lol_gold_path)

lol_kills_path = '../input/leagueoflegends/kills.csv'

lol_kills = pd.read_csv(lol_kills_path)

lol_League_path = '../input/leagueoflegends/LeagueofLegends.csv'

lol_League = pd.read_csv(lol_League_path)

lol_match_path = '../input/leagueoflegends/matchinfo.csv'

lol_match = pd.read_csv(lol_match_path)

lol_monster_path = '../input/leagueoflegends/monsters.csv'

lol_monster = pd.read_csv(lol_monster_path)

lol_structure_path = '../input/leagueoflegends/structures.csv'

lol_structure = pd.read_csv(lol_structure_path)
lol_Complete = pd.DataFrame()
missinglist = pd.DataFrame()

missinglist['Source Title'] = ['Ban', 'Gold', 'Kills', 'League', 'Match Info', 'Monster', 'Structure']

missinglist['Addresses Missing'] = [lol_ban['Address'].isnull().sum(), lol_gold['Address'].isnull().sum(), lol_kills['Address'].isnull().sum(), lol_League['Address'].isnull().sum(), lol_match['Address'].isnull().sum(), lol_monster['Address'].isnull().sum(), lol_structure['Address'].isnull().sum()]

missinglist
lol_gold1 = pd.DataFrame()

lol_gold2 = pd.DataFrame()

lol_gold2['Address'] = lol_League['Address']

lol_gold2['rResult'] = lol_League['rResult']

lol_gold2['bResult'] = lol_League['bResult']

lol_gold1 = lol_gold.loc[(lol_gold['Type'] == "goldblue") | (lol_gold['Type'] == "goldred")]

lol_gold1 = pd.melt(lol_gold1, ['Address', 'Type'], var_name = 'Minute')

lol_gold1['Minute'] = lol_gold1['Minute'].map(lambda x: x.lstrip('min_'))

lol_gold1['Minute'] = lol_gold1['Minute'].astype('int')

lol_gold1 = pd.merge(lol_gold1, lol_gold2, on = 'Address', how = 'left')

lol_gold1['win'] = 'no'

lol_gold1.loc[((lol_gold1['Type']=='goldblue') & (lol_gold1['bResult']==1)) |

             ((lol_gold1['Type']=='goldred') & (lol_gold1['rResult']==1)), 

              'win'] = 'yes'

lol_gold1 = lol_gold1.dropna()

lol_gold1
plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1.5)

graph = sns.scatterplot(x='Minute',y='value', hue = 'win', data=lol_gold1)

graph.set_xticks(np.arange(0,95), minor=True)

graph.set_title("Gold vs Time", fontsize=10)

graph.set_xlabel("Minutes", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
list(lol_gold['Type'].unique())
lol_goldTop = lol_gold.loc[(lol_gold['Type'] == "goldblueTop") | (lol_gold['Type'] == "goldredTop")]

lol_goldTop = pd.melt(lol_goldTop, ['Address', 'Type'], var_name = 'Minute')

lol_goldTop['Minute'] = lol_goldTop['Minute'].map(lambda x: x.lstrip('min_'))

lol_goldTop['Minute'] = lol_goldTop['Minute'].astype('int')

lol_goldTop = pd.merge(lol_goldTop, lol_gold2, on = 'Address', how = 'left')

lol_goldTop['win'] = 'no'

lol_goldTop.loc[((lol_goldTop['Type']=='goldblueTop') & (lol_goldTop['bResult']==1)) |

             ((lol_goldTop['Type']=='goldredTop') & (lol_goldTop['rResult']==1)), 

              'win'] = 'yes'



plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1.5)

graph = sns.scatterplot(x='Minute',y='value', hue= 'win', data=lol_goldTop)

graph.set_xticks(np.arange(0,95), minor=True)

graph.set_title("Top Lane Gold vs Time", fontsize=10)

graph.set_xlabel("Minutes", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
lol_goldJungle = lol_gold.loc[(lol_gold['Type'] == "goldblueJungle") | (lol_gold['Type'] == "goldredJungle")]

lol_goldJungle = pd.melt(lol_goldJungle, ['Address', 'Type'], var_name = 'Minute')

lol_goldJungle['Minute'] = lol_goldJungle['Minute'].map(lambda x: x.lstrip('min_'))

lol_goldJungle['Minute'] = lol_goldJungle['Minute'].astype('int')

lol_goldJungle = pd.merge(lol_goldJungle, lol_gold2, on = 'Address', how = 'left')

lol_goldJungle['win'] = 'no'

lol_goldJungle.loc[((lol_goldJungle['Type']=='goldblueJungle') & (lol_goldJungle['bResult']==1)) |

             ((lol_goldJungle['Type']=='goldredJungle') & (lol_goldJungle['rResult']==1)), 

              'win'] = 'yes'



plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1.5)

graph = sns.scatterplot(x='Minute',y='value', hue= 'win', data=lol_goldJungle)

graph.set_xticks(np.arange(0,95), minor=True)

graph.set_title("Jungle Gold vs Time", fontsize=10)

graph.set_xlabel("Minutes", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
lol_goldMiddle = lol_gold.loc[(lol_gold['Type'] == "goldblueMiddle") | (lol_gold['Type'] == "goldredMiddle")]

lol_goldMiddle = pd.melt(lol_goldMiddle, ['Address', 'Type'], var_name = 'Minute')

lol_goldMiddle['Minute'] = lol_goldMiddle['Minute'].map(lambda x: x.lstrip('min_'))

lol_goldMiddle['Minute'] = lol_goldMiddle['Minute'].astype('int')

lol_goldMiddle = pd.merge(lol_goldMiddle, lol_gold2, on = 'Address', how = 'left')

lol_goldMiddle['win'] = 'no'

lol_goldMiddle.loc[((lol_goldMiddle['Type']=='goldblueMiddle') & (lol_goldMiddle['bResult']==1)) |

             ((lol_goldMiddle['Type']=='goldredMiddle') & (lol_goldMiddle['rResult']==1)), 

              'win'] = 'yes'



plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1.5)

graph = sns.scatterplot(x='Minute',y='value', hue= 'win', data=lol_goldMiddle)

graph.set_xticks(np.arange(0,95), minor=True)

graph.set_title("Middle Lane Gold vs Time", fontsize=10)

graph.set_xlabel("Minutes", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
lol_goldADC = lol_gold.loc[(lol_gold['Type'] == "goldblueADC") | (lol_gold['Type'] == "goldredADC")]

lol_goldADC = pd.melt(lol_goldADC, ['Address', 'Type'], var_name = 'Minute')

lol_goldADC['Minute'] = lol_goldADC['Minute'].map(lambda x: x.lstrip('min_'))

lol_goldADC['Minute'] = lol_goldADC['Minute'].astype('int')

lol_goldADC = pd.merge(lol_goldADC, lol_gold2, on = 'Address', how = 'left')

lol_goldADC['win'] = 'no'

lol_goldADC.loc[((lol_goldADC['Type']=='goldblueADC') & (lol_goldADC['bResult']==1)) |

             ((lol_goldADC['Type']=='goldredADC') & (lol_goldADC['rResult']==1)), 

              'win'] = 'yes'



plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1.5)

graph = sns.scatterplot(x='Minute',y='value', hue= 'win', data=lol_goldADC)

graph.set_xticks(np.arange(0,95), minor=True)

graph.set_title("ADC Player Gold vs Time", fontsize=10)

graph.set_xlabel("Minutes", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
lol_goldSupport = lol_gold.loc[(lol_gold['Type'] == "goldblueSupport") | (lol_gold['Type'] == "goldredSupport")]

lol_goldSupport = pd.melt(lol_goldSupport, ['Address', 'Type'], var_name = 'Minute')

lol_goldSupport['Minute'] = lol_goldSupport['Minute'].map(lambda x: x.lstrip('min_'))

lol_goldSupport['Minute'] = lol_goldSupport['Minute'].astype('int')

lol_goldSupport = pd.merge(lol_goldSupport, lol_gold2, on = 'Address', how = 'left')

lol_goldSupport['win'] = 'no'

lol_goldSupport.loc[((lol_goldSupport['Type']=='goldblueSupport') & (lol_goldSupport['bResult']==1)) |

             ((lol_goldSupport['Type']=='goldredSupport') & (lol_goldSupport['rResult']==1)), 

              'win'] = 'yes'



plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1.5)

graph = sns.scatterplot(x='Minute',y='value', hue= 'win', data=lol_goldSupport)

graph.set_xticks(np.arange(0,95), minor=True)

graph.set_title("Support Player Gold vs Time", fontsize=10)

graph.set_xlabel("Minutes", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
lol_goldJungle = lol_gold.loc[(lol_gold['Type'] == "goldblueJungle") | (lol_gold['Type'] == "goldredJungle")]

lol_goldJungle = pd.melt(lol_goldJungle, ['Address', 'Type'], var_name = 'Minute')

lol_goldJungle['Minute'] = lol_goldJungle['Minute'].map(lambda x: x.lstrip('min_'))

lol_goldJungle['Minute'] = lol_goldJungle['Minute'].astype('int')

lol_goldJungle = pd.merge(lol_goldJungle, lol_gold2, on = 'Address', how = 'left')

lol_goldJungle['win'] = 'no'

lol_goldJungle.loc[((lol_goldJungle['Type']=='goldblueJungle') & (lol_goldJungle['bResult']==1)) |

             ((lol_goldJungle['Type']=='goldredJungle') & (lol_goldJungle['rResult']==1)), 

              'win'] = 'yes'



plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1.5)

graph = sns.scatterplot(x='Minute',y='value', hue= 'win', data=lol_goldJungle)

graph.set_xlim(0,6)

graph.set_ylim(400,3000)

graph.set_title("Jungle Gold vs Time", fontsize=10)

graph.set_xlabel("Minutes", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
redBan = pd.DataFrame()

blueBan = pd.DataFrame()

lol_Complete['Address'] = lol_ban['Address']

blueBan = lol_ban.loc[lol_ban['Team'] == 'blueBans']

redBan = lol_ban.loc[lol_ban['Team'] == 'redBans']

blueBan = blueBan.rename({'ban_1':'Blue_Ban_1', 'ban_2':'Blue_Ban_2', 'ban_3':'Blue_Ban_3', 'ban_4':'Blue_Ban_4', 'ban_5':'Blue_Ban_5'}, axis=1)

redBan = redBan.rename({'ban_1':'Red_Ban_1', 'ban_2':'Red_Ban_2', 'ban_3':'Red_Ban_3', 'ban_4':'Red_Ban_4', 'ban_5':'Red_Ban_5'}, axis=1)

redBan = redBan[['Address','Red_Ban_1', 'Red_Ban_2', 'Red_Ban_3', 'Red_Ban_4', 'Red_Ban_5']]

blueBan = blueBan[['Address','Blue_Ban_1', 'Blue_Ban_2', 'Blue_Ban_3', 'Blue_Ban_4', 'Blue_Ban_5']]

lol_Complete = pd.merge(lol_Complete, blueBan, on = 'Address', how = 'left')

lol_Complete = pd.merge(lol_Complete, redBan, on = 'Address', how = 'left')

lol_Complete.head()
ban1 = lol_ban.ban_1.value_counts() + lol_ban.ban_2.value_counts() + lol_ban.ban_3.value_counts() + lol_ban.ban_4.value_counts() + lol_ban.ban_5.value_counts()

ban1 = ban1.dropna()

ban1 = ban1.sort_values(ascending = False).iloc[:15]

plt.figure(figsize=(75,25))

sns.set_context("paper", font_scale=3.5)

g = sns.barplot(x = ban1.index, y = ban1)

plt.title("Banned Heros")

plt.ylabel("Number of Bans")

plt.xlabel("Heros")
top_killer = lol_kills['Killer'].value_counts()

top_killer = top_killer.dropna()

top_killer = top_killer.sort_values(ascending = False).iloc[:15]

plt.figure(figsize=(75,25))

sns.set_context("paper", font_scale=3.5)

g = sns.barplot(x = top_killer.index, y = top_killer)

plt.title("Top Killers")

plt.ylabel("Number of Kills")

plt.xlabel("Players")
top_victim = lol_kills['Victim'].value_counts()

top_victim = top_victim.dropna()

top_victim = top_victim.sort_values(ascending = False).iloc[:15]

plt.figure(figsize=(75,25))

sns.set_context("paper", font_scale=3.5)

g = sns.barplot(x = top_victim.index, y = top_victim)

plt.title("Most Deaths")

plt.ylabel("Number of Deaths")

plt.xlabel("Players")
lol_kills
cond_map = pd.DataFrame()

cond_map['Team'] = lol_kills['Team']

cond_map['x_pos'] = lol_kills['x_pos']

cond_map['y_pos'] = lol_kills['y_pos']

cond_map['x_pos'] = pd.to_numeric(cond_map['x_pos'], errors = 'coerce')

cond_map['y_pos'] = pd.to_numeric(cond_map['y_pos'], errors = 'coerce')

cond_map.dropna()

sns.set_context("paper", font_scale=1.5)

plt.figure(figsize=(15,15))

plt.xlim([0,15000])

plt.ylim([0,15000])

g = sns.scatterplot(x = 'x_pos', y = 'y_pos', hue = 'Team', data = cond_map)
red_map = pd.DataFrame()

red_map = lol_kills.loc[lol_kills['Team'] == 'rKills']

#red_map['x_pos'] = lol_kills['x_pos']

#red_map['y_pos'] = lol_kills['y_pos']

red_map['x_pos'] = pd.to_numeric(red_map['x_pos'], errors = 'coerce')

red_map['y_pos'] = pd.to_numeric(red_map['y_pos'], errors = 'coerce')

red_map.dropna()

sns.set_context("paper", font_scale=1.5)

plt.figure(figsize=(15,15))

plt.xlim([0,15000])

plt.ylim([0,15000])

g = sns.scatterplot(x = 'x_pos', y = 'y_pos',color = ['red'], data = red_map)
blue_map = pd.DataFrame()

blue_map = lol_kills.loc[lol_kills['Team'] == 'bKills']

#red_map['x_pos'] = lol_kills['x_pos']

#red_map['y_pos'] = lol_kills['y_pos']

blue_map['x_pos'] = pd.to_numeric(blue_map['x_pos'], errors = 'coerce')

blue_map['y_pos'] = pd.to_numeric(blue_map['y_pos'], errors = 'coerce')

blue_map.dropna()

sns.set_context("paper", font_scale=1.5)

plt.figure(figsize=(15,15))

plt.xlim([0,15000])

plt.ylim([0,15000])

g = sns.scatterplot(x = 'x_pos', y = 'y_pos', data = blue_map)
countassist = lol_kills['Assist_1'].value_counts() + lol_kills['Assist_2'].value_counts() + lol_kills['Assist_3'].value_counts() + lol_kills['Assist_4'].value_counts()

countassist = countassist.sort_values(ascending = False).iloc[:15]

sns.set_context("paper", font_scale=4.5)

plt.figure(figsize=(75,25))

g = sns.barplot(x = countassist.index, y = countassist)



plt.title("Players With the Most Assists", fontsize=30)

plt.ylabel("Number of Assists", fontsize=30)

plt.xlabel("Players", fontsize=30)
countTop = lol_League['redADCChamp'].value_counts() + lol_League['blueADCChamp'].value_counts()

countTop = countTop.sort_values(ascending = False).iloc[:15]

sns.set_context("paper", font_scale=5)

plt.figure(figsize=(75,25))

g = sns.barplot(x = countTop.index, y = countTop)



plt.title("ADC Champions", fontsize=30)

plt.ylabel("Number of Appearances", fontsize=30)

plt.xlabel("Champions", fontsize=30)
countTop = lol_League['redSupportChamp'].value_counts() + lol_League['blueSupportChamp'].value_counts()

countTop = countTop.sort_values(ascending = False).iloc[:15]

sns.set_context("paper", font_scale=5)

plt.figure(figsize=(75,25))

g = sns.barplot(x = countTop.index, y = countTop)



plt.title("Support Champions", fontsize=30)

plt.ylabel("Number of Appearances", fontsize=30)

plt.xlabel("Champions", fontsize=30)
countTop = lol_League['redJungleChamp'].value_counts() + lol_League['blueJungleChamp'].value_counts()

countTop = countTop.sort_values(ascending = False).iloc[:15]

sns.set_context("paper", font_scale=5)

plt.figure(figsize=(75,25))

g = sns.barplot(x = countTop.index, y = countTop)



plt.title("Jungle Champions", fontsize=30)

plt.ylabel("Number of Appearances", fontsize=30)

plt.xlabel("Champions", fontsize=30)
countTop = lol_League['redMiddleChamp'].value_counts() + lol_League['blueMiddleChamp'].value_counts()

countTop = countTop.sort_values(ascending = False).iloc[:15]

sns.set_context("paper", font_scale=5)

plt.figure(figsize=(75,25))

g = sns.barplot(x = countTop.index, y = countTop)



plt.title("Middle Champions", fontsize=30)

plt.ylabel("Number of Appearances", fontsize=30)

plt.xlabel("Champions", fontsize=30)
countTop = lol_League['redTopChamp'].value_counts() + lol_League['blueTopChamp'].value_counts()

countTop = countTop.sort_values(ascending = False).iloc[:15]

sns.set_context("paper", font_scale=5)

plt.figure(figsize=(75,25))

g = sns.barplot(x = countTop.index, y = countTop)



plt.title("Top Lane Champions", fontsize=30)

plt.ylabel("Number of Appearances", fontsize=30)

plt.xlabel("Champions", fontsize=30)
lol_player = pd.DataFrame()

lol_bJung = pd.DataFrame()

lol_bMid = pd.DataFrame()

lol_bSup = pd.DataFrame()

lol_bAd = pd.DataFrame()

lol_rTop = pd.DataFrame()

lol_rJung = pd.DataFrame()

lol_rMid = pd.DataFrame()

lol_rSup = pd.DataFrame()

lol_rAd = pd.DataFrame()



lol_player['Address'] = lol_League['Address']

lol_bJung['Address'] = lol_League['Address']

lol_bMid['Address'] = lol_League['Address']

lol_bSup ['Address'] = lol_League['Address']

lol_bAd['Address'] = lol_League['Address']

lol_rTop['Address'] = lol_League['Address']

lol_rJung['Address'] = lol_League['Address']

lol_rMid['Address'] = lol_League['Address']

lol_rSup['Address'] = lol_League['Address']

lol_rAd['Address'] = lol_League['Address']



lol_player['Name'] = lol_League["blueTop"]

lol_bJung['Name'] = lol_League["blueJungle"]

lol_bMid['Name'] = lol_League["blueMiddle"]

lol_bSup['Name'] = lol_League["blueSupport"]

lol_bAd['Name'] = lol_League["blueADC"]

lol_rTop['Name'] = lol_League["redTop"]

lol_rJung['Name'] = lol_League["redJungle"]

lol_rMid['Name'] = lol_League["redMiddle"]

lol_rSup['Name'] = lol_League["redSupport"]

lol_rAd['Name'] = lol_League["redADC"]



lol_player['Role'] = 'Top'

lol_bJung['Role'] = 'Jungle'

lol_bMid['Role'] = 'Middle'

lol_bSup ['Role'] ='Support'

lol_bAd['Role'] = 'ADC'

lol_rTop['Role'] = 'Top'

lol_rJung['Role'] = 'Jungle'

lol_rMid['Role'] = 'Middle'

lol_rSup['Role'] = 'Support'

lol_rAd['Role'] = 'ADC'



lol_player['Result'] = np.where(lol_League['rResult']==1, 0, 1 )

lol_bJung['Result'] = np.where(lol_League['rResult']==1, 0, 1 )

lol_bMid['Result'] = np.where(lol_League['rResult']==1, 0, 1 )

lol_bSup ['Result'] = np.where(lol_League['rResult']==1, 0, 1 )

lol_bAd['Result'] = np.where(lol_League['rResult']==1, 0, 1 )

lol_rTop['Result'] = np.where(lol_League['rResult']==1, 1, 0 )

lol_rJung['Result'] = np.where(lol_League['rResult']==1, 1, 0 )

lol_rMid['Result'] = np.where(lol_League['rResult']==1, 1, 0 )

lol_rSup['Result'] = np.where(lol_League['rResult']==1,1, 0 )

lol_rAd['Result'] = np.where(lol_League['rResult']==1, 1, 0 )



lol_player['Champ'] = lol_League["blueTopChamp"]

lol_bJung['Champ'] = lol_League["blueJungleChamp"]

lol_bMid['Champ'] = lol_League["blueMiddleChamp"]

lol_bSup['Champ'] = lol_League["blueSupportChamp"]

lol_bAd['Champ'] = lol_League["blueADCChamp"]

lol_rTop['Champ'] = lol_League["redTopChamp"]

lol_rJung['Champ'] = lol_League["redJungleChamp"]

lol_rMid['Champ'] = lol_League["redMiddleChamp"]

lol_rSup['Champ'] = lol_League["redSupportChamp"]

lol_rAd['Champ'] = lol_League["redADCChamp"]



lol_player = lol_player.append(lol_bJung, ignore_index = True)

lol_player = lol_player.append(lol_bMid, ignore_index = True)

lol_player = lol_player.append(lol_bSup, ignore_index = True)

lol_player = lol_player.append(lol_bAd, ignore_index = True)

lol_player = lol_player.append(lol_rTop, ignore_index = True)

lol_player = lol_player.append(lol_rJung, ignore_index = True)

lol_player = lol_player.append(lol_rMid, ignore_index = True)

lol_player = lol_player.append(lol_rSup, ignore_index = True)

lol_player = lol_player.append(lol_rAd, ignore_index = True)



lol_player['NameCount'] = 0

lol_player['NameCount'] = lol_player.groupby('Name')['NameCount'].transform('count')

lol_player['WinAvg'] = lol_player.groupby('Name')['Result'].transform('mean')



lol_player['ChampCount'] = 0

lol_player['ChampCount'] = lol_player.groupby('Champ')['ChampCount'].transform('count')

lol_player['ChampWinAvg'] = lol_player.groupby('Champ')['Result'].transform('mean')

lol_player.head()
playerTop = pd.DataFrame()

playerTop['Name'] = lol_player['Name']

playerTop['NameCount'] = lol_player['NameCount']

playerTop['Role'] = lol_player['Role']

playerTop['WinAvg'] = lol_player['WinAvg']

playerTop = playerTop.sort_values(by = 'NameCount', ascending = False)

playerTop = playerTop.dropna()

playerTop = playerTop.drop_duplicates()

playerTop = playerTop.sort_values(by = 'NameCount', ascending = False).iloc[:15]

sns.set_context("paper", font_scale=5)

plt.figure(figsize=(75,25))

g = sns.barplot(x = 'Name', y = 'NameCount' , hue = 'Role', data = playerTop)

plt.title("Top Lane Champions", fontsize=30)

plt.ylabel("Number of Appearances", fontsize=30)

plt.xlabel("Champions", fontsize=30)
playerTop1 = pd.DataFrame()

playerTop1['Name'] = lol_player['Name']

playerTop1['NameCount'] = lol_player['NameCount']

playerTop1['Role'] = lol_player['Role']

playerTop1['WinAvg'] = lol_player['WinAvg']

playerTop1 = playerTop1.sort_values(by = 'NameCount', ascending = False)

playerTop1 = playerTop1.dropna()

playerTop1 = playerTop1.drop_duplicates()

indexNames = playerTop1[ playerTop1['NameCount'] < 15 ].index

playerTop1.drop(indexNames, inplace = True)

playerTop1 = playerTop1.sort_values(by = 'WinAvg', ascending = False).iloc[:25]

sns.set_context("paper", font_scale = 3.5)

plt.figure(figsize=(75,25))

g = sns.barplot(x = 'Name', y = 'WinAvg', data = playerTop1)

plt.title("Players With The Highest Winning Average Over 15 Games", fontsize=30)

plt.ylabel("Win Percentage", fontsize=30)

plt.xlabel("Players", fontsize=30)
playerTop1 = pd.DataFrame()

playerTop1['Champ'] = lol_player['Champ']

playerTop1['ChampCount'] = lol_player['ChampCount']

playerTop1['ChampWinAvg'] = lol_player['ChampWinAvg']

playerTop1 = playerTop1.sort_values(by = 'ChampCount', ascending = False)

playerTop1 = playerTop1.dropna()

playerTop1 = playerTop1.drop_duplicates()

indexNames = playerTop1[ playerTop1['ChampCount'] < 15 ].index

playerTop1.drop(indexNames, inplace = True)

playerTop1 = playerTop1.sort_values(by = 'ChampWinAvg', ascending = False).iloc[:25]

sns.set_context("paper", font_scale = 3.5)

plt.figure(figsize=(75,25))

g = sns.barplot(x = 'Champ', y = 'ChampWinAvg', data = playerTop1)

plt.title("Champions With The Highest Winning Average Over 15 Games", fontsize=30)

plt.ylabel("Win Percentage", fontsize=30)

plt.xlabel("Champions", fontsize=30)
lol_Imp = pd.DataFrame()

lol_waste = pd.DataFrame()

lol_waste['Year'] = lol_League['Year'].astype('str')

lol_Imp['Address'] = lol_League['Address']

lol_Imp['Match Title'] = lol_League['League'].str.cat(lol_League['Season'],sep = ' ').str.cat(lol_waste['Year'],sep = ' ')

lol_Imp['Type'] = lol_League['Type']

lol_Imp['Blue Team'] = lol_League['blueTeamTag']

lol_Imp['Red Team'] = lol_League['redTeamTag']

lol_Imp['Winning Team'] = np.where(lol_League['bResult']==1, 'BLUE', 'RED')

lol_Imp['Game Length'] = lol_League['gamelength']



lol_Imp['Blue Top'] = lol_League['blueTop']

lol_Imp['Blue Top NameCount'] = 0

lol_Imp['Blue Top NameCount'] = lol_League.groupby('blueTop').transform('count')

lol_Imp['Blue Top WinAvg'] = lol_League.groupby('blueTop')['bResult'].transform('mean')



lol_Imp['Blue Jungle'] = lol_League['blueJungle']

lol_Imp['Blue Jungle NameCount'] = 0

lol_Imp['Blue Jungle NameCount'] = lol_League.groupby('blueJungle').transform('count')

lol_Imp['Blue Jungle WinAvg'] = lol_League.groupby('blueJungle')['bResult'].transform('mean')



lol_Imp['Blue Middle'] = lol_League['blueMiddle']

lol_Imp['Blue Middle NameCount'] = 0

lol_Imp['Blue Middle NameCount'] = lol_League.groupby('blueMiddle').transform('count')

lol_Imp['Blue Middle WinAvg'] = lol_League.groupby('blueMiddle')['bResult'].transform('mean')



lol_Imp['Blue ADC'] = lol_League['blueADC']

lol_Imp['Blue ADC NameCount'] = 0

lol_Imp['Blue ADC NameCount'] = lol_League.groupby('blueADC').transform('count')

lol_Imp['Blue ADC WinAvg'] = lol_League.groupby('blueADC')['bResult'].transform('mean')



lol_Imp['Blue Support'] = lol_League['blueSupport']

lol_Imp['Blue Support NameCount'] = 0

lol_Imp['Blue Support NameCount'] = lol_League.groupby('blueSupport').transform('count')

lol_Imp['Blue Support WinAvg'] = lol_League.groupby('blueSupport')['bResult'].transform('mean')



lol_Imp['Blue Top Champ'] = lol_League['blueTopChamp']

lol_Imp['Blue Top ChampCount'] = 0

lol_Imp['Blue Top ChampCount'] = lol_League.groupby('blueTopChamp').transform('count')

lol_Imp['Blue Top ChampAvg'] = lol_League.groupby('blueTopChamp')['bResult'].transform('mean')



lol_Imp['Blue Jungle Champ'] = lol_League['blueJungleChamp']

lol_Imp['Blue Jungle ChampCount'] = 0

lol_Imp['Blue Jungle ChampCount'] = lol_League.groupby('blueJungleChamp').transform('count')

lol_Imp['Blue Jungle ChampAvg'] = lol_League.groupby('blueJungleChamp')['bResult'].transform('mean')



lol_Imp['Blue Middle Champ'] = lol_League['blueMiddleChamp']

lol_Imp['Blue Middle ChampCount'] = 0

lol_Imp['Blue Middle ChampCount'] = lol_League.groupby('blueMiddleChamp').transform('count')

lol_Imp['Blue Middle ChampAvg'] = lol_League.groupby('blueMiddleChamp')['bResult'].transform('mean')



lol_Imp['Blue ADC Champ'] = lol_League['blueADCChamp']

lol_Imp['Blue ADC ChampCount'] = 0

lol_Imp['Blue ADC ChampCount'] = lol_League.groupby('blueADCChamp').transform('count')

lol_Imp['Blue ADC ChampAvg'] = lol_League.groupby('blueADCChamp')['bResult'].transform('mean')



lol_Imp['Blue Support Champ'] = lol_League['blueSupportChamp']

lol_Imp['Blue Support ChampCount'] = 0

lol_Imp['Blue Support ChampCount'] = lol_League.groupby('blueSupportChamp').transform('count')

lol_Imp['Blue Support ChampAvg'] = lol_League.groupby('blueSupportChamp')['bResult'].transform('mean')



lol_Imp['Red Top'] = lol_League['redTop']

lol_Imp['Red Top NameCount'] = 0

lol_Imp['Red Top NameCount'] = lol_League.groupby('redTop').transform('count')

lol_Imp['Red Top WinAvg'] = lol_League.groupby('redTop')['rResult'].transform('mean')



lol_Imp['Red Jungle'] = lol_League['redJungle']

lol_Imp['Red Jungle NameCount'] = 0

lol_Imp['Red Jungle NameCount'] = lol_League.groupby('redJungle').transform('count')

lol_Imp['Red Jungle WinAvg'] = lol_League.groupby('redJungle')['rResult'].transform('mean')



lol_Imp['Red Middle'] = lol_League['redMiddle']

lol_Imp['Red Middle NameCount'] = 0

lol_Imp['Red Middle NameCount'] = lol_League.groupby('redMiddle').transform('count')

lol_Imp['Red Middle WinAvg'] = lol_League.groupby('redMiddle')['rResult'].transform('mean')



lol_Imp['Red ADC'] = lol_League['redADC']

lol_Imp['Red ADC NameCount'] = 0

lol_Imp['Red ADC NameCount'] = lol_League.groupby('redADC').transform('count')

lol_Imp['Red ADC WinAvg'] = lol_League.groupby('redADC')['rResult'].transform('mean')



lol_Imp['Red Support'] = lol_League['redSupport']

lol_Imp['Red Support NameCount'] = 0

lol_Imp['Red Support NameCount'] = lol_League.groupby('redSupport').transform('count')

lol_Imp['Red Support WinAvg'] = lol_League.groupby('redSupport')['rResult'].transform('mean')



lol_Imp['Red Top Champ'] = lol_League['redTopChamp']

lol_Imp['Red Top ChampCount'] = 0

lol_Imp['Red Top ChampCount'] = lol_League.groupby('redTopChamp').transform('count')

lol_Imp['Red Top ChampAvg'] = lol_League.groupby('redTopChamp')['rResult'].transform('mean')



lol_Imp['Red Jungle Champ'] = lol_League['redJungleChamp']

lol_Imp['Red Jungle ChampCount'] = 0

lol_Imp['Red Jungle ChampCount'] = lol_League.groupby('redJungleChamp').transform('count')

lol_Imp['Red Jungle ChampAvg'] = lol_League.groupby('redJungleChamp')['rResult'].transform('mean')



lol_Imp['Red Middle Champ'] = lol_League['redMiddleChamp']

lol_Imp['Red Middle ChampCount'] = 0

lol_Imp['Red Middle ChampCount'] = lol_League.groupby('redMiddleChamp').transform('count')

lol_Imp['Red Middle ChampAvg'] = lol_League.groupby('redMiddleChamp')['rResult'].transform('mean')



lol_Imp['Red ADC Champ'] = lol_League['redADCChamp']

lol_Imp['Red ADC ChampCount'] = 0

lol_Imp['Red ADC ChampCount'] = lol_League.groupby('redADCChamp').transform('count')

lol_Imp['Red ADC ChampAvg'] = lol_League.groupby('redADCChamp')['rResult'].transform('mean')



lol_Imp['Red Support Champ'] = lol_League['redSupportChamp']

lol_Imp['Red Support ChampCount'] = 0

lol_Imp['Red Support ChampCount'] = lol_League.groupby('redSupportChamp').transform('count')

lol_Imp['Red Support ChampAvg'] = lol_League.groupby('redSupportChamp')['rResult'].transform('mean')

lol_Imp.head()
lol_Complete = pd.merge(lol_Imp, lol_Complete, on = 'Address', how = 'left')

lol_Complete.head()
plt.hist(x = lol_Complete['Game Length'], bins = 95)

sns.set_context("paper", font_scale=2)

plt.title("Time per Match")

plt.ylabel("Number of Matches")

plt.xlabel("Length of Match")
teamcount = lol_Complete['Red Team'].value_counts() + lol_Complete['Blue Team'].value_counts()

teamcount = teamcount.sort_values(ascending = False).iloc[:15]

plt.figure(figsize=(105,30))

g = sns.barplot(x = teamcount.index, y = teamcount)

sns.set_context("paper", font_scale=5)

plt.title("Time Each Team Competes")

plt.ylabel("Number of Appearances")

plt.xlabel("Teams")
lol_match.head()
Lcount = lol_match.League.value_counts()

Lcount = Lcount.sort_values(ascending = False)

plt.figure(figsize=(75,25))

g = sns.barplot(x = Lcount.index, y = Lcount)

plt.title("Matches in Each League")

plt.ylabel("Number of Matches")

plt.xlabel("Leagues")
Ycount = lol_match.Year.value_counts()

Ycount = Ycount.sort_values(ascending = False)

plt.figure(figsize=(20,10))

g = sns.barplot(x = Ycount.index, y = Ycount)

plt.title("Matches in Each Year")

plt.ylabel("Number of Matches")

plt.xlabel("Years")
Tcount = lol_match.Type.value_counts()

Tcount = Tcount.sort_values(ascending = False)

sns.set_context("paper", font_scale=1.5)

plt.figure(figsize=(8,8))

g = sns.barplot(x = Tcount.index, y = Tcount)

plt.title("Matches in Each Match Type")

plt.ylabel("Number of Matches")

plt.xlabel("Leagues")
monster1 = pd.DataFrame()

monster2 = pd.DataFrame()

monster1 = lol_monster

monster2['Address'] = lol_League['Address']

monster2['rResult'] = lol_League['rResult']

monster1 = pd.merge(monster1, monster2, on = 'Address', how = 'left')

monster1['win'] = 'no'

monster1.loc[(((monster1['Team']=='bDragons') | (monster1['Team']=='bHeralds') | (monster1['Team']=='bBarons'))& (monster1['rResult']==0)) 

              |

             (((monster1['Team']=='rDragons') | (monster1['Team']=='rHeralds') | (monster1['Team']=='rBarons') )& (monster1['rResult']==1)), 

              'win'] = 'yes'

monster1.loc[((monster1['Team']=='bDragons') | (monster1['Team']=='bHeralds') | (monster1['Team']=='bBarons')), 'Team'] = 'blue'

monster1.loc[((monster1['Team']=='rDragons') | (monster1['Team']=='rHeralds') | (monster1['Team']=='rBarons')), 'Team'] = 'red'

monster1
plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1)

graph = sns.violinplot( x=monster1["Team"], y=monster1["Time"] )

graph.set_title("Team Killed Monster", fontsize=10)

graph.set_xlabel("Team", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1)

graph = sns.violinplot( x=monster1["Type"], y=monster1["Time"] )

graph.set_title("Time Team Killed Monster", fontsize=10)

graph.set_xlabel("Monster Type", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1)

graph = sns.violinplot( x=monster1["win"], y=monster1["Time"] )

graph.set_title("Time Team Killed Monster", fontsize=10)

graph.set_xlabel("Win", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
monster1.head()
MTcount = lol_monster.Type.value_counts()

MTcount = MTcount.sort_values(ascending = False)

plt.figure(figsize=(75,25))

g = sns.barplot(x = MTcount.index, y = MTcount)

plt.title("Matches in Each League")

plt.ylabel("Number of Matches")

plt.xlabel("Leagues")
list(lol_structure['Team'].unique())
structure1 = pd.DataFrame()

structure2 = pd.DataFrame()

structure1 = lol_structure

structure2['Address'] = lol_League['Address']

structure2['rResult'] = lol_League['rResult']

structure1 = pd.merge(structure1, structure2, on = 'Address', how = 'left')

structure1['win'] = 'no'

structure1.loc[(((structure1['Team']=='bTowers') | (structure1['Team']=='bInhibs'))& (structure1['rResult']==0)) 

              |

             (((structure1['Team']=='rTowers') | (structure1['Team']=='rInhibs'))& (structure1['rResult']==1)), 

              'win'] = 'yes'

structure1.loc[((structure1['Team']=='bTowers') | (structure1['Team']=='bInhibs')), 'Team'] = 'blue'

structure1.loc[((structure1['Team']=='rTowers') | (structure1['Team']=='rInhibs')), 'Team'] = 'red'

structure1 = structure1.dropna()

structure1
plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1)

graph = sns.violinplot( x=structure1["Type"], y=structure1["Time"] )

graph.set_title("Turret Type Destruction at Times", fontsize=10)

graph.set_xlabel("Turret Type", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
plt.figure(figsize=(10,4))

sns.set_context("paper", font_scale=1)

graph = sns.violinplot( x=structure1["win"], y=structure1["Time"] )

graph.set_title("Winning Team vs Losing Team Destruction Spread", fontsize=10)

graph.set_xlabel("Win", fontsize=14)

graph.set_ylabel("Time", fontsize=14)
justmon = pd.DataFrame()

justmon['Address'] = lol_monster['Address']

justmon['Monster Type'] = lol_monster['Type']

lol_Complete = pd.merge(lol_Complete, justmon, on = 'Address', how = 'left')
list(lol_Complete)
lol_Comp_Cat = pd.DataFrame()

lol_Comp_Cat["Match Title"] = lol_Complete["Match Title"].astype('category')

lol_Comp_Cat["Type"] = lol_Complete["Type"].astype('category')

lol_Comp_Cat["Blue Team"] = lol_Complete["Blue Team"].astype('category')

lol_Comp_Cat["Red Team"] = lol_Complete["Red Team"].astype('category')

lol_Comp_Cat["Winning Team"] = lol_Complete["Winning Team"].astype('category')

lol_Comp_Cat['Game Length'] = lol_Complete['Game Length']

lol_Comp_Cat['Blue Top'] = lol_Complete['Blue Top'].astype('category')

lol_Comp_Cat['Blue Jungle'] = lol_Complete['Blue Jungle'].astype('category')

lol_Comp_Cat['Blue Middle'] = lol_Complete['Blue Middle'].astype('category')

lol_Comp_Cat['Blue ADC'] = lol_Complete['Blue ADC'].astype('category')

lol_Comp_Cat['Blue Support'] = lol_Complete['Blue Support'].astype('category')

lol_Comp_Cat['Blue Top Champ'] = lol_Complete['Blue Top Champ'].astype('category')

lol_Comp_Cat['Blue Jungle Champ'] = lol_Complete['Blue Jungle Champ'].astype('category')

lol_Comp_Cat['Blue Middle Champ'] = lol_Complete['Blue Middle Champ'].astype('category')

lol_Comp_Cat['Blue ADC Champ'] = lol_Complete['Blue ADC Champ'].astype('category')

lol_Comp_Cat['Blue Support Champ'] = lol_Complete['Blue Support Champ'].astype('category')

lol_Comp_Cat['Red Top'] = lol_Complete['Red Top'].astype('category')

lol_Comp_Cat['Red Jungle'] = lol_Complete['Red Jungle'].astype('category')

lol_Comp_Cat['Red Middle'] = lol_Complete['Red Middle'].astype('category')

lol_Comp_Cat['Red ADC'] = lol_Complete['Red ADC'].astype('category')

lol_Comp_Cat['Red Support'] = lol_Complete['Red Support'].astype('category')

lol_Comp_Cat['Red Top Champ'] = lol_Complete['Red Top Champ'].astype('category')

lol_Comp_Cat['Red Jungle Champ'] = lol_Complete['Red Jungle Champ'].astype('category')

lol_Comp_Cat['Red Middle Champ'] = lol_Complete['Red Middle Champ'].astype('category')

lol_Comp_Cat['Red ADC Champ'] = lol_Complete['Red ADC Champ'].astype('category')

lol_Comp_Cat['Red Support Champ'] = lol_Complete['Red Support Champ'].astype('category')

lol_Comp_Cat["Blue_Ban_1"] = lol_Complete["Blue_Ban_1"].astype('category')

lol_Comp_Cat["Blue_Ban_2"] = lol_Complete["Blue_Ban_2"].astype('category')

lol_Comp_Cat["Blue_Ban_3"] = lol_Complete["Blue_Ban_3"].astype('category')

lol_Comp_Cat["Blue_Ban_4"] = lol_Complete["Blue_Ban_4"].astype('category')

lol_Comp_Cat["Blue_Ban_5"] = lol_Complete["Blue_Ban_5"].astype('category')

lol_Comp_Cat["Red_Ban_1"] = lol_Complete["Red_Ban_1"].astype('category')

lol_Comp_Cat["Red_Ban_2"] = lol_Complete["Red_Ban_2"].astype('category')

lol_Comp_Cat["Red_Ban_3"] = lol_Complete["Red_Ban_3"].astype('category')

lol_Comp_Cat["Red_Ban_4"] = lol_Complete["Red_Ban_4"].astype('category')

lol_Comp_Cat["Red_Ban_5"] = lol_Complete["Red_Ban_5"].astype('category')

#lol_Comp_Cat["Monster Type"] = lol_Complete["Monster Type"].astype('category')





lol_Comp_Cat["Match Title"] = lol_Comp_Cat["Match Title"].cat.codes

lol_Comp_Cat["Type"] = lol_Comp_Cat["Type"].cat.codes

lol_Comp_Cat["Blue Team"] = lol_Comp_Cat["Blue Team"].cat.codes

lol_Comp_Cat["Red Team"] = lol_Comp_Cat["Red Team"].cat.codes

lol_Comp_Cat["Winning Team"] = lol_Comp_Cat["Winning Team"].cat.codes

lol_Comp_Cat['Blue Top'] = lol_Comp_Cat['Blue Top'].cat.codes

lol_Comp_Cat['Blue Top NameCount'] = lol_Imp['Blue Top NameCount']

lol_Comp_Cat['Blue Top WinAvg'] = lol_Imp['Blue Top WinAvg']

lol_Comp_Cat['Blue Jungle'] = lol_Comp_Cat['Blue Jungle'].cat.codes

lol_Comp_Cat['Blue Jungle NameCount'] = lol_Imp['Blue Jungle NameCount']

lol_Comp_Cat['Blue Jungle WinAvg'] = lol_Imp['Blue Jungle WinAvg']

lol_Comp_Cat['Blue Middle'] = lol_Comp_Cat['Blue Middle'].cat.codes

lol_Comp_Cat['Blue Middle NameCount'] = lol_Imp['Blue Middle NameCount']

lol_Comp_Cat['Blue Middle WinAvg'] = lol_Imp['Blue Middle WinAvg']

lol_Comp_Cat['Blue ADC'] = lol_Comp_Cat['Blue ADC'].cat.codes

lol_Comp_Cat['Blue ADC NameCount'] = lol_Imp['Blue ADC NameCount']

lol_Comp_Cat['Blue ADC WinAvg'] = lol_Imp['Blue ADC WinAvg']

lol_Comp_Cat['Blue Support'] = lol_Comp_Cat['Blue Support'].cat.codes

lol_Comp_Cat['Blue Support NameCount'] = lol_Imp['Blue Support NameCount']

lol_Comp_Cat['Blue Support WinAvg'] = lol_Imp['Blue Support WinAvg']

lol_Comp_Cat['Blue Top Champ'] = lol_Comp_Cat['Blue Top Champ'].cat.codes

lol_Comp_Cat['Blue Top ChampCount'] = lol_Imp['Blue Top ChampCount']

lol_Comp_Cat['Blue Top ChampAvg'] = lol_Imp['Blue Top ChampAvg']

lol_Comp_Cat['Blue Jungle Champ'] = lol_Comp_Cat['Blue Jungle Champ'].cat.codes

lol_Comp_Cat['Blue Jungle ChampCount'] = lol_Imp['Blue Jungle ChampCount']

lol_Comp_Cat['Blue Jungle ChampAvg'] = lol_Imp['Blue Jungle ChampAvg']

lol_Comp_Cat['Blue Middle Champ'] = lol_Comp_Cat['Blue Middle Champ'].cat.codes

lol_Comp_Cat['Blue Middle ChampCount'] = lol_Imp['Blue Middle ChampCount']

lol_Comp_Cat['Blue Middle ChampAvg'] = lol_Imp['Blue Middle ChampAvg']

lol_Comp_Cat['Blue ADC Champ'] = lol_Comp_Cat['Blue ADC Champ'].cat.codes

lol_Comp_Cat['Blue ADC ChampCount'] = lol_Imp['Blue ADC ChampCount']

lol_Comp_Cat['Blue ADC ChampAvg'] = lol_Imp['Blue ADC ChampAvg']

lol_Comp_Cat['Blue Support Champ'] = lol_Comp_Cat['Blue Support Champ'].cat.codes

lol_Comp_Cat['Blue Support ChampCount'] = lol_Imp['Blue Support ChampCount']

lol_Comp_Cat['Blue Support ChampAvg'] = lol_Imp['Blue Support ChampAvg']

lol_Comp_Cat['Red Top'] = lol_Comp_Cat['Red Top'].cat.codes

lol_Comp_Cat['Red Top NameCount'] = lol_Imp['Red Top NameCount']

lol_Comp_Cat['Red Top WinAvg'] = lol_Imp['Red Top WinAvg']

lol_Comp_Cat['Red Jungle'] = lol_Comp_Cat['Red Jungle'].cat.codes

lol_Comp_Cat['Red Jungle NameCount'] = lol_Imp['Red Jungle NameCount']

lol_Comp_Cat['Red Jungle WinAvg'] = lol_Imp['Red Jungle WinAvg']

lol_Comp_Cat['Red Middle'] = lol_Comp_Cat['Red Middle'].cat.codes

lol_Comp_Cat['Red Middle NameCount'] = lol_Imp['Red Middle NameCount']

lol_Comp_Cat['Red Middle WinAvg'] = lol_Imp['Red Middle WinAvg']

lol_Comp_Cat['Red ADC'] = lol_Comp_Cat['Red ADC'].cat.codes

lol_Comp_Cat['Red ADC NameCount'] = lol_Imp['Red ADC NameCount']

lol_Comp_Cat['Red ADC WinAvg'] = lol_Imp['Red ADC WinAvg']

lol_Comp_Cat['Red Support'] = lol_Comp_Cat['Red Support'].cat.codes

lol_Comp_Cat['Red Support NameCount'] = lol_Imp['Red Support NameCount']

lol_Comp_Cat['Red Support WinAvg'] = lol_Imp['Red Support WinAvg']

lol_Comp_Cat['Red Top Champ'] = lol_Comp_Cat['Red Top Champ'].cat.codes

lol_Comp_Cat['Red Top ChampCount'] = lol_Imp['Red Top ChampCount']

lol_Comp_Cat['Red Top ChampAvg'] = lol_Imp['Red Top ChampAvg']

lol_Comp_Cat['Red Jungle Champ'] = lol_Comp_Cat['Red Jungle Champ'].cat.codes

lol_Comp_Cat['Red Jungle ChampCount'] = lol_Imp['Red Jungle ChampCount']

lol_Comp_Cat['Red Jungle ChampAvg'] = lol_Imp['Red Jungle ChampAvg']

lol_Comp_Cat['Red Middle Champ'] = lol_Comp_Cat['Red Middle Champ'].cat.codes

lol_Comp_Cat['Red Middle ChampCount'] = lol_Imp['Red Middle ChampCount']

lol_Comp_Cat['Red Middle ChampAvg'] = lol_Imp['Red Middle ChampAvg']

lol_Comp_Cat['Red ADC Champ'] = lol_Comp_Cat['Red ADC Champ'].cat.codes

lol_Comp_Cat['Red ADC ChampCount'] = lol_Imp['Red ADC ChampCount']

lol_Comp_Cat['Red ADC ChampAvg'] = lol_Imp['Red ADC ChampAvg']

lol_Comp_Cat['Red Support Champ'] = lol_Comp_Cat['Red Support Champ'].cat.codes

lol_Comp_Cat['Red Support ChampCount'] = lol_Imp['Red Support ChampCount']

lol_Comp_Cat['Red Support ChampAvg'] = lol_Imp['Red Support ChampAvg']

lol_Comp_Cat["Blue_Ban_1"] = lol_Comp_Cat["Blue_Ban_1"].cat.codes

lol_Comp_Cat["Blue_Ban_2"] = lol_Comp_Cat["Blue_Ban_2"].cat.codes

lol_Comp_Cat["Blue_Ban_3"] = lol_Comp_Cat["Blue_Ban_3"].cat.codes

lol_Comp_Cat["Blue_Ban_4"] = lol_Comp_Cat["Blue_Ban_4"].cat.codes

lol_Comp_Cat["Blue_Ban_5"] = lol_Comp_Cat["Blue_Ban_5"].cat.codes

lol_Comp_Cat["Red_Ban_1"] = lol_Comp_Cat["Red_Ban_1"].cat.codes

lol_Comp_Cat["Red_Ban_2"] = lol_Comp_Cat["Red_Ban_2"].cat.codes

lol_Comp_Cat["Red_Ban_3"] = lol_Comp_Cat["Red_Ban_3"].cat.codes

lol_Comp_Cat["Red_Ban_4"] = lol_Comp_Cat["Red_Ban_4"].cat.codes

lol_Comp_Cat["Red_Ban_5"] = lol_Comp_Cat["Red_Ban_5"].cat.codes

#lol_Comp_Cat["Monster Type"] = lol_Comp_Cat["Monster Type"].cat.codes

lol_Comp_Cat = lol_Comp_Cat.dropna()

lol_Comp_Cat = lol_Comp_Cat.drop_duplicates()

lol_Comp_Cat.head()
lol_Complete.head()
y = lol_Comp_Cat['Winning Team']

features = ['Match Title', 'Type', 'Blue Team', 'Red Team', 'Game Length', 'Blue Top', 'Blue Jungle', 'Blue Middle', 'Blue ADC', 'Blue Support', 'Blue Top Champ', 'Blue Jungle Champ', 'Blue Middle Champ', 'Blue ADC Champ', 'Blue Support Champ',

 'Red Top', 'Red Jungle', 'Red Middle', 'Red ADC', 'Red Support', 'Red Top Champ', 'Red Jungle Champ', 'Red Middle Champ', 'Red ADC Champ', 'Red Support Champ', 'Blue_Ban_1', 'Blue_Ban_2', 'Blue_Ban_3', 'Blue_Ban_4',  'Red_Ban_1', 'Red_Ban_2',

 'Red_Ban_3', 'Red_Ban_4', 'Red Middle WinAvg']

#, 'Red Jungle ChampAvg', 'Red ADC ChampCount', 'Red ADC ChampAvg', 'Red Middle ChampAvg', 'Red Middle ChampCount', 'Red Top ChampAvg','Red Top ChampCount', 'Blue Support ChampAvg', 'Blue Support ChampCount', 'Blue Jungle ChampCount','Blue Jungle ChampAvg',

#'Blue Middle ChampAvg','Blue Top ChampAvg','Blue Top ChampCount', 'Red Support WinAvg','Red ADC NameCount','Red Jungle WinAvg', 'Red Jungle NameCount',  'Red Top WinAvg', 'Red Top NameCount','Blue Support WinAvg',   'Blue Jungle WinAvg',

#'Blue Jungle NameCount','Blue Middle WinAvg', 'Blue Middle NameCount','Blue Top WinAvg',  'Blue Top NameCount', 'Red_Ban_5', 'Red Support ChampAvg', 'Red Support ChampCount', 'Red Jungle ChampCount', 'Blue ADC ChampCount',

#'Blue Middle ChampCount', 'Red Support NameCount','Red ADC WinAvg','Red Middle NameCount','Blue Support NameCount','Blue ADC WinAvg','Blue ADC NameCount','Blue_Ban_5'

#, 'Blue ADC ChampAvg'

x = lol_Comp_Cat[features]

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

basic_model = DecisionTreeRegressor()

basic_model.fit(train_x, train_y)

val_predictions = basic_model.predict(val_x)

print("Basic Decision Tree Regressor \nMean Absolute Error:", mean_absolute_error(val_y, val_predictions))
def get_mae(max_leaf_nodes, train_x, val_X, train_y, val_y):

    leaf_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    leaf_model.fit(train_x, train_y)

    preds_val = leaf_model.predict(val_x)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))
forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_x, train_y)

forest_preds = forest_model.predict(val_x)

print("Random Forest Model \nMean Absolute Error: ",mean_absolute_error(val_y, forest_preds))
my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),

                              ('model', RandomForestRegressor(n_estimators=50,random_state=0))])

points_CV = -1 * cross_val_score(my_pipeline, x, y, cv=5, scoring = 'neg_mean_absolute_error')

print("Using Cross Validation..\nNow Printing Mean Absolute Error points:\n", points_CV)
print("Cross Validation Model \nAverage Mean Absolute Error: ", points_CV.mean())
xgbr_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

xgbr_model.fit(train_x, train_y, 

             early_stopping_rounds=5, 

             eval_set=[(val_x, val_y)], 

             verbose=False)

predictionsXBGR = xgbr_model.predict(val_x)

print("XCBRegressor Model")

print("Mean Absolute Error: " + str(mean_absolute_error(predictionsXBGR, val_y)))
perm = PermutationImportance(basic_model, random_state=1).fit(val_x, val_y)

eli5.show_weights(perm, feature_names = val_x.columns.tolist())