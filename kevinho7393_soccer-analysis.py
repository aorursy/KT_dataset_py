# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import sqlite3

import pandas as pd

from dateutil.parser import parse

from datetime import datetime

import xlrd

from pandas import ExcelWriter

from numbers import Number

import matplotlib.pyplot as plt



conn = sqlite3.connect('../input/database.sqlite')

c = conn.cursor()



c.execute("SELECT name FROM sqlite_master WHERE type='table';")

print(c.fetchall())



player = pd.read_sql_query("""

                           select * from Player;

                           """, 

                           conn)

Player_Attributes = pd.read_sql_query("""

                           select * from Player_Attributes;

                           """, 

                           conn)



df_players_combine = pd.merge(left=Player_Attributes, right=player, how='left', left_on='player_api_id', right_on='player_api_id')

df_players_combine['date'] = pd.to_datetime(df_players_combine['date'])



# find player age

t = pd.to_datetime(df_players_combine['date'])

d = pd.to_datetime(df_players_combine['birthday'])



df_players_combine['age'] = (t - d).astype('<m8[Y]')





# find stats of all players who played for Arsenal

matches = pd.read_sql_query("""

                           select * from Match;

                           """, 

                           conn)

                           

teams = pd.read_sql_query("""

                           select * from Team;

                           """, 

                           conn)



df_match_home = pd.merge(left=matches, right=teams, how='left', left_on='home_team_api_id', right_on='team_api_id')

df_match_home.head()
df_match_home.rename(columns={'team_api_id':'home_team_id', 'team_fifa_api_id': 'home_team_fifa_id', 'team_long_name':'home_team', 'team_short_name': 'home_team_short'}, inplace=True)



df_match_combined = pd.merge(left=df_match_home, right=teams, how='left', left_on='away_team_api_id', right_on='team_api_id')



df_match_combined.rename(columns={'team_api_id':'away_team_id', 'team_fifa_api_id': 'away_team_fifa_id', 'team_long_name':'away_team', 'team_short_name': 'away_team_short'}, inplace=True)

df_ArsHomeMatch_filter = df_match_combined[(df_match_combined['home_team'] == 'Arsenal')]

df_ArsAwayMatch_filter = df_match_combined[(df_match_combined['away_team'] == 'Arsenal')]

df_ArsHomePlayers = df_ArsHomeMatch_filter.iloc[:, 55:66]

df_ArsAwayPlayers = df_ArsAwayMatch_filter.iloc[:, 66:77]

home_players_list = pd.unique(df_ArsHomePlayers.values.ravel())

home_players_list = home_players_list.tolist()

away_players_list = pd.unique(df_ArsAwayPlayers.values.ravel())

away_players_list = away_players_list.tolist()

Ars_players_list = home_players_list + away_players_list

Ars_players_list = list(set(Ars_players_list))



df_player_filter = df_players_combine[df_players_combine['player_api_id'].isin(Ars_players_list)]

df_player_filter = df_player_filter.sort_values(['age'], ascending=[1])



#plot player age vs rating

x = df_player_filter['age']

y = df_player_filter['overall_rating']

plt.plot(x, y)

plt.show()
df_player_filter.head()