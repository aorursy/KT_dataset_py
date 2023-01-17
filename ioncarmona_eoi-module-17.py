#Import all the libraries we need to undertake the exercise:

import sqlite3

import pandas as pd

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from bokeh.plotting import figure, ColumnDataSource, show

from bokeh.models import HoverTool

from bokeh.io import output_notebook

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as NP

import math

#Connection with the Database:

%matplotlib inline

output_notebook()



database = '../input/database.sqlite'

conn = sqlite3.connect(database)



#List all tables we found in the database

query = "SELECT name as Tablas FROM sqlite_master WHERE type='table';"

pd.read_sql(query, conn)
#Understanding and reading the tables:

with sqlite3.connect('../input/database.sqlite') as con:

    player_attributes_all = pd.read_sql_query("SELECT * from Player_Attributes", con)

    player_all = pd.read_sql_query("SELECT * from Player", con)

    match_all = pd.read_sql_query("SELECT * from Match", con)

    league_all = pd.read_sql_query("SELECT * from League", con)

    country_all = pd.read_sql_query("SELECT * from Country", con)

    team_all = pd.read_sql_query("SELECT * from Team", con)

    team_attributes_all = pd.read_sql_query("SELECT * from Team_Attributes", con)

con.close()
player_all.head()
player_attributes_all.head()
match=match_all.copy()

match.drop(match.columns[NP.r_[ 11:115]], axis=1, inplace=True)
match.info()

match.head()
def winner(row):

    if row['home_team_goal'] > row['away_team_goal']:

        val = row["home_team_api_id"]

    elif row['home_team_goal'] < row['away_team_goal']:

        val = row["away_team_api_id"]

    else:

        val = None

    return val
match['winnerID'] = match.apply(winner, axis=1)

#we only keep the match that the team have won

match=match.dropna(thresh=12) 

#convert it to an integer (int)

match['winnerID'].astype(int)
topWinner = match[['match_api_id','winnerID']].groupby(['winnerID'])['match_api_id'].count().nlargest(1000).reset_index(name='NumberWimMatch')

topWinner = pd.merge(topWinner, team_all,  how='left', left_on=['winnerID'], right_on = ['team_api_id'])

topWinner = topWinner[['winnerID','NumberWimMatch','team_long_name']]

print(topWinner)
victoryBarcelona = match.loc[match['winnerID']==8634]
evolution = victoryBarcelona[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

import matplotlib.pyplot as plt

plt.plot(evolution['season'].str[:4], evolution['NumberWimMatch'], marker='o', linestyle='-', color='r')

plt.show()
victoryMadrid = match.loc[match['winnerID']==8633]

evolution = victoryMadrid[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

plt.plot(evolution['season'].str[:4], evolution['NumberWimMatch'], marker='o', linestyle='-', color='r')

plt.show()
victoryAthleticBilbao = match.loc[match['winnerID']==8315]

evolution = victoryAthleticBilbao[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

import matplotlib.pyplot as plt

plt.plot(evolution['season'].str[:4], evolution['NumberWimMatch'], marker='o', linestyle='-', color='r')

plt.show()
victoryRealSociedad = match.loc[match['winnerID']==8560]

evolution = victoryRealSociedad[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

plt.plot(evolution['season'].str[:4], evolution['NumberWimMatch'], marker='o', linestyle='-', color='r')

plt.show()