# Imports

%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import sqlite3

import numpy as np

import missingno as ms

from numpy import random



# Cargamos datos

database = '../input/database.sqlite'

con = sqlite3.connect(database)

    

# Listamos las tablas existentes

query = "SELECT name as Tablas FROM sqlite_master WHERE type='table';"

pd.read_sql(query, con)
query1 = "SELECT * FROM Player_Attributes;"

d1 = pd.read_sql(query1, con)

d1.head()
d1.info()
query2 = "SELECT * FROM Player;"

d2 = pd.read_sql(query2, con)

d2.head()
d2.info()
query3 = "SELECT * FROM Match;"

d3 = pd.read_sql(query3, con)

d3.head()
d3.info()
query4 = "SELECT * FROM League;"

d4 = pd.read_sql(query4, con)

d4.head()
d4.info()
query5 = "SELECT * FROM Country;"

d5 = pd.read_sql(query5, con)

d5.head()
d5.info()
query6 = "SELECT * FROM Team;"

d6 = pd.read_sql(query6, con)

d6.head()
d6.info()
query7 = "SELECT * FROM Team_attributes;"

d7 = pd.read_sql(query7, con)

d7.head()
d7.info()
ms.bar(d1)
top_defenders = pd.read_sql_query("select   substr(a.date,1,4) as date ,b.player_name,avg(a.defensive_work_rate+a.reactions+a.interceptions+a.positioning+a.marking+a.standing_tackle+a.sliding_tackle) as media FROM Player_Attributes a inner join (SELECT a.player_api_id,p.player_name,avg(a.defensive_work_rate+a.reactions+a.interceptions+a.positioning+a.marking+a.standing_tackle+a.sliding_tackle) as media FROM Player_Attributes a INNER JOIN Player p 	on a.player_api_id=p.player_api_id group by  a.player_api_id,p.player_name order by avg(a.defensive_work_rate+a.reactions+a.interceptions+a.positioning+a.marking+a.standing_tackle+a.sliding_tackle) desc limit 30) b on a.player_api_id=b.player_api_id Group by substr(a.date,1,4),b.player_name order by substr(a.date,1,4),avg(a.defensive_work_rate+a.reactions+a.interceptions+a.positioning+a.marking+a.standing_tackle+a.sliding_tackle) desc;",con)
plt.plot(top_defenders['date'].str[:4], top_defenders['media'], marker='o', linestyle=' ', color='b', label = "Puntuación Obtenida")

plt.title("30 Mejores Jugadores Defensivos por Año")

plt.xlabel("Año") 

plt.ylabel("Puntuación media")

plt.legend(loc='lower right')

plt.show()
top_defenders.groupby(['player_name'])['media'].mean().sort_values(ascending=False)