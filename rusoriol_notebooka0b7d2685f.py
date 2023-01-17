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



%matplotlib inline

output_notebook()



database = '../input/database.sqlite'

conn = sqlite3.connect(database)



# list all tables

query = "SELECT name as Tablas FROM sqlite_master WHERE type='table';"

pd.read_sql(query, conn)
#leyendo todas las tablas

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
match_all.head()
match=match_all.copy()

match.drop(match.columns[NP.r_[ 11:115]], axis=1, inplace=True)
match.info()

match.head()
def ganador(row):

    if row['home_team_goal'] > row['away_team_goal']:

        val = row["home_team_api_id"]

    elif row['home_team_goal'] < row['away_team_goal']:

        val = row["away_team_api_id"]

    else:

        val = None

    return val

match['winnerID'] = match.apply(ganador, axis=1)

#nos quedamos con solo los partidos que ha habido un ganador

match=match.dropna(thresh=12) 

#convertimos a entero

match['winnerID'].astype(int)
topGanadores = match[['match_api_id','winnerID']].groupby(['winnerID'])['match_api_id'].count().nlargest(10).reset_index(name='NumberWimMatch')

topGanadores=pd.merge(topGanadores, team_all,  how='left', left_on=['winnerID'], right_on = ['team_api_id'])

topGanadores=topGanadores[['winnerID','NumberWimMatch','team_long_name']]

print(topGanadores)
partidoGanadosBarcelona=match.loc[match['winnerID']==8634]
evolucion = partidoGanadosBarcelona[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

import matplotlib.pyplot as plt

plt.plot(evolucion['season'].str[:4], evolucion['NumberWimMatch'], marker='o', linestyle='-', color='r')

plt.show()
partidoGanadosMadrid=match.loc[match['winnerID']==8633]

evolucion = partidoGanadosMadrid[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

plt.plot(evolucion['season'].str[:4], evolucion['NumberWimMatch'], marker='o', linestyle='-', color='r')

plt.show()