import pandas as pd 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
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



#Insertamos la base de datos en la variable database.

database = '../input/database.sqlite'

conn = sqlite3.connect(database)



#Listamos todas las tablas que la componen.

query = "SELECT name as Tablas FROM sqlite_master WHERE type='table';"

pd.read_sql(query, conn)
#Guardamos cada una de las tablas en una variable.

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

#Nos quedamos con solo los partidos que ha habido un ganador

match=match.dropna(thresh=12) 

#Convertimos a entero

match['winnerID'].astype(int)
topGanadores = match[['match_api_id','winnerID']].groupby(['winnerID'])['match_api_id'].count().nlargest(10).reset_index(name='NumberWimMatch')

topGanadores=pd.merge(topGanadores, team_all,  how='left', left_on=['winnerID'], right_on = ['team_api_id'])

topGanadores=topGanadores[['winnerID','NumberWimMatch','team_long_name']]

print(topGanadores)
#Partidos ganados del FC Barcelona.

partidoGanadosBarcelona=match.loc[match['winnerID']==8634]

partidoGanadosMadrid=match.loc[match['winnerID']==8634]

evolucion = partidoGanadosBarcelona[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

plt.plot(evolucion['season'].str[:4], evolucion['NumberWimMatch'], marker='o', linestyle='-', color='b')



#Partidos ganados del Real Madrid.

partidoGanadosMadrid=match.loc[match['winnerID']==8633]

evolucion = partidoGanadosMadrid[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

plt.plot(evolucion['season'].str[:4], evolucion['NumberWimMatch'], marker='o', linestyle='-', color='grey')



plt.show()

query = """SELECT * FROM Player_Attributes a

           INNER JOIN (SELECT player_name, player_api_id AS p_id FROM Player) b ON a.player_api_id = b.p_id;"""



drop_cols = ['id','player_fifa_api_id','date','preferred_foot',

             'attacking_work_rate','defensive_work_rate']



players = pd.read_sql(query, conn)

players['date'] = pd.to_datetime(players['date'])

players = players[players.date > pd.datetime(2015,1,1)]

players = players[~players.overall_rating.isnull()].sort_values('date', ascending=False)

players = players.drop_duplicates(subset='player_api_id', keep='first')

players = players.drop(drop_cols, axis=1)



players.info()
players = players.sort_values('overall_rating', ascending=False)

best_players = players[['player_api_id','player_name']].head(20)

ids = tuple(best_players.player_api_id.unique())
query = '''SELECT player_api_id, date, overall_rating

           FROM Player_Attributes WHERE player_api_id in %s''' % (ids,)
evolution = pd.read_sql(query, conn)

evolution = pd.merge(evolution, best_players)

evolution['year'] = evolution.date.str[:4].apply(int)

evolution = evolution.groupby(['year','player_api_id','player_name']).overall_rating.mean()

evolution = evolution.reset_index()



#Comprobamos que lo hemos hecho bien.

evolution.head(10)
#Evolución de Messi

messi=evolution.loc[evolution['player_name']=="Lionel Messi"]

messi
#Evolución de Cristiano

cr7=evolution.loc[evolution['player_name']=="Cristiano Ronaldo"]

cr7


#Comparamos ahora las evoluciones de ambos jugadores.

plt.plot(messi["year"], messi['overall_rating'], marker='o', linestyle='-', color='b')

plt.plot(cr7["year"], cr7['overall_rating'], marker='o', linestyle='-', color='grey')

plt.show()
print("Evolución de los partidos ganados por temporada")

evolucion = partidoGanadosBarcelona[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

plt.plot(evolucion['season'].str[:4], evolucion['NumberWimMatch'], marker='o', linestyle='-', color='b')

evolucion = partidoGanadosMadrid[['match_api_id','season']].groupby(['season'])['match_api_id'].count().reset_index(name='NumberWimMatch')

plt.plot(evolucion['season'].str[:4], evolucion['NumberWimMatch'], marker='o', linestyle='-', color='grey')

plt.show()



print("Evolución de las puntuaciones generales de cada jugador por temporada")

plt.plot(messi["year"], messi['overall_rating'], marker='o', linestyle='-', color='b')

plt.plot(cr7["year"], cr7['overall_rating'], marker='o', linestyle='-', color='grey')

plt.show()
