# import all libraries and dependencies for dataframe

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta



# import all libraries and dependencies for data visualization

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [8,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1) 

sns.set(style='darkgrid')

import matplotlib.ticker as ticker

import matplotlib.ticker as plticker
# Reading the file 



path = '../input/pro-kabaddi-2019/'

file = path + 'Player_Stat_Season 7.xlsx'
# Reading the Team_leaderboard file on which Analysis needs to be done



df_Player_Total_points = pd.read_excel(file,'Total Points')



df_Player_Total_points.head()
df_Player_Total_points.info()
df_Player_Total_points['Avg Player Points'] = df_Player_Total_points['Points']/df_Player_Total_points['Matches']

df_Player_Total_points = df_Player_Total_points.drop(['Points','Matches'],axis=1)

df_Player_Total_points = df_Player_Total_points.sort_values(by='Avg Player Points',ascending=False)
df_Player_Raid_points = pd.read_excel(file,'Raid Points')



#df_Player_Raid_points.head(2)

df_Player_Raid_points['Avg Player Raid Points'] = df_Player_Raid_points['Points']/df_Player_Raid_points['Matches']

df_Player_Raid_points = df_Player_Raid_points.drop(['Points','Matches'],axis=1)

df_Player_Raid_points = df_Player_Raid_points.sort_values(by='Avg Player Raid Points',ascending=False)
df_Player_Tackle_points = pd.read_excel(file,'Tackle Points')



df_Player_Tackle_points['Avg Player Tackle Points'] = df_Player_Tackle_points['Points']/df_Player_Tackle_points['Matches']

df_Player_Tackle_points = df_Player_Tackle_points.drop(['Points','Matches'],axis=1)

df_Player_Tackle_points = df_Player_Tackle_points.sort_values(by='Avg Player Tackle Points',ascending=False)
plt.subplots(figsize = (12,8))

ax = sns.barplot(x = 'Player', y = 'Avg Player Points',data=df_Player_Total_points)

plt.title("Avg Player Points of Player in S7", fontsize = 18)

plt.ylabel("Avg Player Points", fontsize = 15)

plt.xlabel("Player",fontsize = 15)

ax=ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
plt.subplots(figsize = (12,8))

ax = sns.barplot(x = 'Player', y = 'Avg Player Raid Points', data=df_Player_Raid_points, edgecolor=(0,0,0), linewidth=0.2)

plt.title("Avg Raid Points of Player in S7", fontsize = 18)

plt.ylabel("Avg Player Raid Points", fontsize = 15)

plt.xlabel("Player",fontsize = 15)

ax=ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
plt.subplots(figsize = (12,8))

ax = sns.barplot(x = 'Player', y = 'Avg Player Tackle Points', data=df_Player_Tackle_points, edgecolor=(0,0,0), linewidth=0.2)

plt.title("Avg Tackle Points of Player in S7", fontsize = 18)

plt.ylabel("Avg Player Tackle Points", fontsize = 15)

plt.xlabel("Player",fontsize = 15)

ax=ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
df_Player_Raid_points = df_Player_Raid_points.sort_values(by='Successful Raid %',ascending=False)

df_Player_Raid_points = df_Player_Raid_points.reset_index(drop=True)

df_Player_Raid_points = df_Player_Raid_points.loc[:5,:]

df_Player_Raid_points
plt.subplots(figsize = (12,8))



ax = sns.barplot(x = 'Player', y = 'Successful Raid %', data=df_Player_Raid_points, edgecolor=(0,0,0), linewidth=0.2)

plt.title("Successful Raid % of Player in S7", fontsize = 18)

plt.ylabel("Successful Raid %", fontsize = 15)

plt.xlabel("Player",fontsize = 15)

ax=ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
df_Player_Tackle_points = df_Player_Tackle_points.sort_values(by='Successful Tackle %',ascending=False)

df_Player_Tackle_points = df_Player_Tackle_points.reset_index(drop=True)

df_Player_Tackle_points = df_Player_Tackle_points.loc[:10,:]

df_Player_Tackle_points
plt.subplots(figsize = (12,8))



ax = sns.barplot(x = 'Player', y = 'Successful Tackle %', data=df_Player_Tackle_points, edgecolor=(0,0,0), linewidth=0.2)

plt.title("Successful Tackle % of Player in S7", fontsize = 18)

plt.ylabel("Successful Tackle %", fontsize = 15)

plt.xlabel("Player",fontsize = 15)

ax=ax.set_xticklabels(ax.get_xticklabels(), rotation=75)