# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# imports

import sqlite3 #database

# TODO: scikit stuff
conn = sqlite3.connect('../input/soccer/database.sqlite')

#Fetching required data tables

player_data = pd.read_sql("SELECT * FROM Player;", conn)

player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)

team_data = pd.read_sql("SELECT * FROM Team;", conn)

match_data = pd.read_sql("SELECT * FROM Match;", conn)
match_data.head()
list(match_data.columns)
match_data[['home_team_api_id',

 'away_team_api_id', 'home_player_1',

 'home_player_2',

 'home_player_3',

 'home_player_4',

 'home_player_5',

 'home_player_6',

 'home_player_7',

 'home_player_8',

 'home_player_9',

 'home_player_10',

 'home_player_11',

 'away_player_1',

 'away_player_2',

 'away_player_3',

 'away_player_4',

 'away_player_5',

 'away_player_6',

 'away_player_7',

 'away_player_8',

 'away_player_9',

 'away_player_10',

 'away_player_11']]
team_data.head()
team_data[team_data['team_api_id'] == 10190]
player_data.columns
player_data[player_data['player_api_id'] == 42231]