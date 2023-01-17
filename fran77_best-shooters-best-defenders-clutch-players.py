# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from matplotlib.pyplot import xticks

from scipy.stats import skew

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/shot_logs.csv')
data.head()
# Best shooters
# Nba shot accuracy

made = data[data.SHOT_RESULT == 'made']['SHOT_RESULT'].count()

accuracy = round(made / len(data), 2)

accuracy
# Shot taken per game

shot_taken = data.groupby(['player_name', 'GAME_ID'])['SHOT_RESULT'].count()

shot_taken.head()
shot_taken = shot_taken.reset_index()

shot_taken = round(shot_taken.groupby('player_name')['SHOT_RESULT'].mean(),2)

shot_taken = shot_taken.reset_index()

shot_taken.columns = ['player_name', 'mean_shot']

shot_taken.head()
shot_taken.mean_shot.mean()
# Player has to take at least 9 shots per game 
players_made = data[data.SHOT_RESULT == 'made'].groupby('player_name')['SHOT_RESULT'].count()

total_shots_player = data.groupby('player_name')['SHOT_RESULT'].count()

accuracy_players = round(players_made / total_shots_player,4) *100
accuracy_players = accuracy_players.reset_index()
accuracy_players.columns = ['player_name', 'accuracy']
accuracy_players['mean_shot_taken']  = shot_taken['mean_shot']
accuracy_filtered_players = accuracy_players[accuracy_players.mean_shot_taken >= 9]
accuracy_filtered_players = accuracy_filtered_players.sort_values(['accuracy'], ascending=False)
accuracy_filtered_players.head()
# These players are all centers, we just have to check their mean distance shot
mean_dist_shot = data.groupby('player_name')['SHOT_DIST'].mean()

mean_dist_shot = mean_dist_shot.reset_index()

mean_dist_shot.columns = ['player_name', 'mean_dist_shot']
accuracy_players['mean_dist_shot'] = mean_dist_shot['mean_dist_shot']
top_accuracy = accuracy_players[accuracy_players.mean_shot_taken >= 9]

top_accuracy = top_accuracy.sort_values(['accuracy'], ascending=False)

top_accuracy = top_accuracy[:10]

top_accuracy = top_accuracy.sort_values(['player_name'])
top_accuracy = top_accuracy.reset_index(drop=True)
top_accuracy
sns.set(rc={'figure.figsize':(8,6)})



fig = plt.figure()

ax1 = fig.add_subplot(111)

g = sns.barplot(x=top_accuracy.player_name, y=top_accuracy.accuracy ,ax=ax1)

g.set_xticklabels(labels = top_accuracy.player_name,  rotation=90)

ax2 = ax1.twinx()

h = sns.lineplot(x="player_name", y="mean_shot_taken", data=top_accuracy, ax=ax2, color="red")

h.set_xticklabels(labels = top_accuracy.player_name,  rotation=90)

ax2.grid(False)

plt.show()
# Best 3 point shooter
shot_3 = data[data.SHOT_DIST >=24]
# 3 point Shot taken per game

shot3_taken = shot_3.groupby(['player_name', 'GAME_ID'])['SHOT_RESULT'].count()

shot3_taken = shot3_taken.reset_index()

shot3_taken = round(shot3_taken.groupby('player_name')['SHOT_RESULT'].mean(),2)

shot3_taken.head()
shot3_taken.mean()
shot3_taken = shot3_taken.reset_index()

shot3_taken.columns = ['player_name', 'mean_shot']

shot3_taken.head()
# Player has to take at least 9 shots per game 

players3_made = shot_3[shot_3.SHOT_RESULT == 'made'].groupby('player_name')['SHOT_RESULT'].count()

total_shots3_player = shot_3.groupby('player_name')['SHOT_RESULT'].count()

accuracy3_players = round(players3_made / total_shots3_player,4) *100
accuracy3_players = accuracy3_players.reset_index()

accuracy3_players.columns = ['player_name', 'accuracy']

accuracy3_players['mean_shot_taken']  = shot3_taken['mean_shot']
top_accuracy3 = accuracy3_players[accuracy3_players.mean_shot_taken > 3]

top_accuracy3 = top_accuracy3.sort_values(['accuracy'], ascending=False)
top_accuracy3 = top_accuracy3[:15]

top_accuracy3 = top_accuracy3.sort_values(['player_name'])

top_accuracy3 = top_accuracy3.reset_index(drop=True)
top_accuracy3
sns.set(rc={'figure.figsize':(8,6)})



fig = plt.figure()

ax1 = fig.add_subplot(111)

g = sns.barplot(x=top_accuracy3.player_name, y=top_accuracy3.accuracy ,ax=ax1)

g.set_xticklabels(labels = top_accuracy3.player_name,  rotation=90)

ax2 = ax1.twinx()

h = sns.lineplot(x="player_name", y="mean_shot_taken", data=top_accuracy3, ax=ax2, color="red")

h.set_xticklabels(labels = top_accuracy3.player_name,  rotation=90)

ax2.grid(False)

plt.show()
# The more shots you take, the more your accuracy should decrease

shot_accuracy3 = accuracy3_players[['accuracy', 'mean_shot_taken']]
shot_accuracy3 = shot_accuracy3.dropna()
shot_accuracy3 = shot_accuracy3.groupby('mean_shot_taken').mean()
shot_accuracy3 = shot_accuracy3.reset_index()
sns.set(rc={'figure.figsize':(10,8)})

sns.scatterplot(x='mean_shot_taken', y= 'accuracy', data=shot_accuracy3)
# Best defenders
data.head()
# Shot taken agianst player per game

shot_defend = data.groupby(['CLOSEST_DEFENDER', 'GAME_ID'])['SHOT_RESULT'].count()

shot_defend.head()
shot_defend = shot_defend.reset_index()

shot_defend = round(shot_defend.groupby('CLOSEST_DEFENDER')['SHOT_RESULT'].mean(),2)

shot_defend = shot_defend.reset_index()

shot_defend.columns = ['CLOSEST_DEFENDER', 'mean_shot']

shot_defend.head()
shot_defend.mean_shot.mean()
shot_defend.mean_shot.max()
# Player has to defend more than 10 shots
defender_miss = data[data.SHOT_RESULT == 'missed'].groupby('CLOSEST_DEFENDER')['SHOT_RESULT'].count()

total_shots_defender = data.groupby('CLOSEST_DEFENDER')['SHOT_RESULT'].count()

accuracy_defenders = round(defender_miss / total_shots_defender,4) *100
accuracy_defenders = accuracy_defenders.reset_index()

accuracy_defenders.columns = ['CLOSEST_DEFENDER', 'accuracy']

accuracy_defenders['mean_shot_defend']  = shot_defend['mean_shot']
top_def = accuracy_defenders[accuracy_defenders.mean_shot_defend > 10]

top_def = top_def.sort_values(['accuracy'], ascending=False)
top_def = top_def[:15]

top_def = top_def.sort_values(['CLOSEST_DEFENDER'])
top_def
sns.set(rc={'figure.figsize':(8,6)})



fig = plt.figure()

ax1 = fig.add_subplot(111)

g = sns.barplot(x=top_def.CLOSEST_DEFENDER, y=top_def.accuracy ,ax=ax1)

g.set_xticklabels(labels = top_def.CLOSEST_DEFENDER,  rotation=90)

ax2 = ax1.twinx()

h = sns.lineplot(x="CLOSEST_DEFENDER", y="mean_shot_defend", data=top_def, ax=ax2, color="red")

h.set_xticklabels(labels = top_def.CLOSEST_DEFENDER,  rotation=90)

ax2.grid(False)

plt.show()
# Clutch players
data.head()
clutch = data[['FINAL_MARGIN', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'PTS_TYPE', 'SHOT_RESULT', 'player_name']]
# 4th quarter

clutch = clutch[clutch.PERIOD == 4]
# Convert to datetime

clutch["GAME_CLOCK_SECONDS"] = pd.to_datetime(data["GAME_CLOCK"], format="%M:%S")
# Convert clock to seconds : for seconds played, 12:00 - x, x as game clock and 12:00 as quarter time

clutch["GAME_CLOCK_SECONDS"] = clutch["GAME_CLOCK_SECONDS"].apply(lambda x: 12*60 - (x.minute * 60 + x.second))
clutch.head()
# Clutch = Last 2 minutes, point difference = 5 pts or less.

quarter_duration = 60*12

clutch = clutch[(np.abs(clutch.FINAL_MARGIN) < 5) & (clutch.GAME_CLOCK_SECONDS >= (quarter_duration - 120))]
clutch.head()
# Shot taken per game

shot_clutch = clutch.groupby(['player_name'])['SHOT_RESULT'].count()

shot_clutch = shot_clutch.reset_index()

shot_clutch.columns = ['player_name', 'shot_taken']

shot_clutch.head()
shot_clutch.shot_taken.mean()
clutch_made = clutch[clutch.SHOT_RESULT == 'made'].groupby('player_name')['SHOT_RESULT'].count()

total_shots_clutch = clutch.groupby('player_name')['SHOT_RESULT'].count()

accuracy_clutch = round(clutch_made / total_shots_clutch,4) *100
accuracy_clutch = accuracy_clutch.reset_index()

accuracy_clutch.columns = ['player_name', 'accuracy']

accuracy_clutch['shot_taken']  = shot_clutch['shot_taken']
accuracy_clutch_players = accuracy_clutch[accuracy_clutch.shot_taken >= 7]

accuracy_clutch_players = accuracy_clutch_players.sort_values(['accuracy'], ascending=False)

accuracy_clutch_players = accuracy_clutch_players[:10]

accuracy_clutch_players = accuracy_clutch_players.sort_values(['player_name'])

accuracy_clutch_players = accuracy_clutch_players.reset_index(drop=True)
accuracy_clutch_players.head()
sns.set(rc={'figure.figsize':(8,6)})



fig = plt.figure()

ax1 = fig.add_subplot(111)

g = sns.barplot(x=accuracy_clutch_players.player_name, y=accuracy_clutch_players.accuracy ,ax=ax1)

g.set_xticklabels(labels = accuracy_clutch_players.player_name,  rotation=90)

ax2 = ax1.twinx()

h = sns.lineplot(x="player_name", y="shot_taken", data=accuracy_clutch_players, ax=ax2, color="red")

h.set_xticklabels(labels = accuracy_clutch_players.player_name,  rotation=90)

ax2.grid(False)

plt.show()
clutch_accuracy = accuracy_clutch.groupby('shot_taken').mean()

clutch_accuracy = clutch_accuracy.reset_index()
clutch_accuracy
sns.set(rc={'figure.figsize':(10,8)})

sns.lineplot(x=clutch_accuracy.shot_taken, y= clutch_accuracy.accuracy)