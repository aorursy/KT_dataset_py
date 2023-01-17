## Importing required libraries

import sqlite3

import pandas as pd

import warnings



warnings.simplefilter("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#path = "../input/"  #Insert path here

#database = path + 'database.sqlite'

database = '/kaggle/input/soccer/database.sqlite'

con = sqlite3.connect(database)
#All Tables in Database

cursor = con.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

print(cursor.fetchall())
sqlite_sequence_data = pd.read_sql("SELECT * FROM sqlite_sequence;", con) 

print(sqlite_sequence_data.shape)

print(sqlite_sequence_data.columns)

sqlite_sequence_data.head()
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", con)

print(player_stats_data.shape)

print(player_stats_data.columns)

player_stats_data.head()
player_data = pd.read_sql("SELECT * FROM Player;", con)

print(player_data.shape)

print(player_data.columns)

player_data.head()
match_data = pd.read_sql("SELECT * FROM Match;", con)

print(match_data.shape)

print(match_data.columns)

match_data.head()
League_data = pd.read_sql("SELECT * FROM League;", con) 

print(League_data.shape)

print(League_data.columns)

League_data.head()
team_data = pd.read_sql("SELECT * FROM Team;", con)

print(team_data.shape)

print(team_data.columns)

team_data.head()
Country_data = pd.read_sql("SELECT * FROM Country;", con) 

print(Country_data.shape)

print(Country_data.columns)

Country_data.head()
Team_Attributes_data = pd.read_sql("SELECT * FROM Team_Attributes;", con) 

print(Team_Attributes_data.shape)

print(Team_Attributes_data.columns)

Team_Attributes_data.head()