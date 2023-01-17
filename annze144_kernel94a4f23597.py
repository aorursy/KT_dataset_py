# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# get files into dataframes that would be used in the process

player_info = pd.read_csv('../input/player_info.csv')

game_shifts = pd.read_csv('../input/game_shifts.csv')

game = pd.read_csv('../input/game.csv')

game_skater_stats=pd.read_csv("../input/game_skater_stats.csv")

team_info=pd.read_csv("../input/team_info.csv")

game_teams_stats=pd.read_csv("../input/game_teams_stats.csv")
# filter only needed players

player_info=player_info[(player_info['lastName']=="Kane") & (player_info['firstName']=="Patrick")| 

           (player_info['lastName']=="Ovechkin") & (player_info['firstName']=="Alex")]

player_info=player_info[['player_id', 'firstName', 'lastName']] 

player_info.head(10)
# filter only needed season for players stats

game_player=game[['game_id','season']] 

game_player=game_player[(game_player['season']==20162017)] 

game_player.head(3)
# filter only needed season for team stats

game_team=game[(game['season']==20172018)] 

game_team=game_team[['game_id','season']] 

game_team.head(3)
#eliminate columns 

game_shifts=game_shifts[['game_id','player_id', 'shift_start', 'shift_end']] 

game_shifts.head(3)
#merge for the shifts per game cnt (1a)

dfmrg = pd.merge(pd.merge(player_info,game_shifts ,on='player_id'),game_player, on = 'game_id') 

dfmrg=dfmrg[['player_id', 'firstName','lastName','game_id','shift_start','shift_end','season']]

dfmrg.head(3)
# Shift cnt per game (1a)

df_shifts_per_game = dfmrg.groupby(['player_id','game_id'])['shift_start'].count().reset_index()

df_shifts_per_game1=df_shifts_per_game.rename(columns={'player_id':'player_id','game_id':'game_id', 'shift_start':'Shift Cnt'})

df_shifts_per_game1.head(10)

#df_shifts_per_game.shape
#avg time per shift (1b)

game_shifts['avg_played'] = game_shifts['shift_end'] - game_shifts['shift_start']

df_time_per_shift = game_shifts.groupby(['game_id','player_id'])['avg_played'].mean().reset_index()/360

#df_time_per_shift['minutes_played'] = df_time_per_shift['seconds_played'] / 60

#df_time_per_shift['hours_played'] = df_time_per_shift['minutes_played'] / 60 

df_time_per_shift.head()
#merge for the shifts per game cnt ???????????

dfmrg = pd.merge(pd.merge(player_info,game_shifts ,on='player_id'),game_player, on = 'game_id') 

dfmrg=dfmrg[['player_id', 'avg_played']]

dfmrg.head(3)
#merge for shots attemps ( 1c)

dfmrg = pd.merge(pd.merge(player_info,game_skater_stats ,on='player_id'),game_player, on = 'game_id') 

dfmrg=dfmrg[['player_id', 'game_id','shots','season']]

dfmrg.head(3)
#shots attemps cnt per game (1c)

df_shot_attempts = dfmrg.groupby(['player_id','game_id'])['shots'].sum().reset_index()

df_shot_attempts.head(5)
#merge for goals per game (1d)

dfmrg = pd.merge(pd.merge(player_info,game_skater_stats ,on='player_id'),game_player, on = 'game_id') 

dfmrg=dfmrg[['player_id', 'game_id','goals','season']]

dfmrg.head(3)
#goals cnt per game (1d)

df_goals_scored = dfmrg.groupby(['player_id','game_id'])['goals'].sum().reset_index()

df_goals_scored.head(5)
#2A shots attemps per season plot

dfmrg = pd.merge(pd.merge(player_info,game_skater_stats ,on='player_id'),game_player, on = 'game_id') 

dfmrg=dfmrg[['player_id', 'season','shots',]]

dfmrg.head(3)
#2A shots attemps per seazon plot

df_shot_attempts = dfmrg.groupby(['player_id'])['shots'].sum().reset_index()

df_shot_attempts.head(5)
#2A shots attemps per seazon plot

df_shot_attempts = dfmrg.groupby(['player_id'])['shots'].sum().plot(kind='bar',legend=True, title='Shots attemps for season')

#df_shot_attempts.head(5)
#2B hits per seazon plot

dfmrg = pd.merge(pd.merge(player_info,game_skater_stats ,on='player_id'),game_player, on = 'game_id') 

dfmrg=dfmrg[['player_id', 'game_id','goals','season']]

#dfmrg.head(3)

df_goals_scored = dfmrg.groupby(['player_id' ])['goals'].sum().plot(kind='bar',legend=True,title='Hits for season')

#df_goals_scored.head(5)
#filtering firm info

team_info=team_info[(team_info['shortName']=='Washington') & (team_info['teamName']=='Capitals')]

team_info.head(3)
#joining for Power Play Percentage(3a)

dfmrg = pd.merge(pd.merge(team_info,game_teams_stats ,on = 'team_id'),game_team, on = 'game_id') 

dfmrg=dfmrg[['team_id','powerPlayGoals','powerPlayOpportunities','season']]

dfmrg.head(3)
#merge for takeways (3b)

dfmrg = pd.merge(pd.merge(team_info,game_teams_stats ,on = 'team_id'),game_team, on = 'game_id') 

dfmrg=dfmrg[['team_id','game_id','giveaways','takeaways','season']]

dfmrg.head(3)
# Average per game (3b)

dfmrg['AveragePerGame'] = (dfmrg['takeaways'] -dfmrg['giveaways']) 

dfmrg.groupby(['team_id', 'game_id'])['AveragePerGame'].mean()

dfmrg.head(3)
#joining for plot team shot atempts for season(4a)

dfmrg = pd.merge(pd.merge(team_info,game_teams_stats ,on = 'team_id'),game_team, on = 'game_id') 

dfmrg=dfmrg[['team_id','game_id','shots','season']]

dfmrg.head(3)
#plot shot attempts(4a)

df_team_shots = dfmrg.groupby(['team_id' ])['shots'].sum().plot(kind='bar',legend=True,title='Shots attempts for season')
#joining for plot team hits atempts for season(4b)

dfmrg = pd.merge(pd.merge(team_info,game_teams_stats ,on = 'team_id'),game_team, on = 'game_id') 

dfmrg=dfmrg[['team_id','game_id','hits','season']]

dfmrg.head(3)
#plot hits(4b)

df_team_hits = dfmrg.groupby(['team_id' ])['hits'].sum().plot(kind='bar',legend=True,title='Total Hits for season')