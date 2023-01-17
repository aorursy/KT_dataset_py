%matplotlib inline





import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import sqlite3

import numpy as np

from numpy import random

import missingno as msno
with sqlite3.connect('../input/database.sqlite') as con:

    #DATOS MAESTROS

    Paises = pd.read_sql_query("SELECT * from Country", con)

    Ligas = pd.read_sql_query("SELECT * from League", con)

    Equipos = pd.read_sql_query("SELECT * from Team", con)

    Atributos_equipos = pd.read_sql_query("SELECT * from Team_Attributes", con)

    Atributos_jugadores = pd.read_sql_query("SELECT * from Player_Attributes", con)

    Entidades= pd.read_sql_query("SELECT* from sqlite_sequence", con)

    #DATOS TRANSACCIONALES

    Partidos = pd.read_sql_query("SELECT * from Match", con)



    

 


Entidades


 #DATOS MAESTROS

Paises_Reg = pd.read_sql_query("SELECT count(*) from Country", con)

Ligas_Reg = pd.read_sql_query("SELECT count (*) from League", con)

Equipos_Reg = pd.read_sql_query("SELECT count (*) from Team", con)

Atributos_equipos_Reg = pd.read_sql_query("SELECT count(*) from Team_Attributes", con)

Atributos_jugadores_Reg = pd.read_sql_query("SELECT count (*) from Player_Attributes", con)

Entidades= pd.read_sql_query("SELECT count (*) from sqlite_sequence", con)

    #DATOS TRANSACCIONALES

Partidos_Reg = pd.read_sql_query("SELECT count (*) from Match", con)



print ("Registros Paises")

print(Paises_Reg )

print ("Registros Ligas")

print(Ligas_Reg)

print ("Registros Equipos")

print(Equipos_Reg)

print ("Registros Atributos_equipos")

print(Atributos_equipos_Reg)

print ("Registros Atributos_jugadores")

print(Atributos_jugadores_Reg)

print ("Registros Partidos")

print(Partidos_Reg)
# Nos centraremos primero en los datos operacionales, los datos de los partidos: 

Partidos.head()
Partidos.tail()
Partidos.info()
msno.bar(Partidos)
Partidos_Equipos= pd.read_sql_query("SELECT league_id,season,date,match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal from Match", con)

Partidos_Equipos.head()
print (Equipos)
print(Ligas)
print(Paises)
#Analizamos el periodo temporal 

Partidos_tiempo_0= pd.read_sql_query("SELECT min(date) from Match", con)

Partidos_tiempo_1= pd.read_sql_query("SELECT max(date) from Match", con)



print(Partidos_tiempo_0)

print(Partidos_tiempo_1)
Possession= pd.read_sql_query("SELECT date, league_id, possession from Match where possession not null", con)

print(Possession)
Possession_null= pd.read_sql_query("SELECT date, league_id, possession from Match where possession is null", con)

print(Possession_null)
Atributos_jugadores.head()
Atributos_jugadores.tail()
Atributos_jugadores.info()
#tenemos más de 180.000 mil registros

#Ahora verificamos la calidad de los datos 

msno.bar(Atributos_jugadores)
Atributos_equipos.head()
Atributos_equipos.tail()
Atributos_equipos.tail()
Atributos_equipos.info()
print(Atributos_equipos)
msno.bar(Atributos_equipos)
Partidos_Equipo= pd.read_sql_query("SELECT name, season, date,  Home, T.team_long_name as Away, home_team_goal, away_team_goal FROM (SELECT c.name, m.season, m.date, m.match_api_id, m.away_team_api_id, t.team_long_name AS Home,  m.home_team_goal, m.away_team_goal  FROM Match AS m INNER JOIN Country AS c ON m.country_id = c.id INNER JOIN Team AS t ON  m.home_team_api_id = t.team_api_id) A INNER JOIN Team AS T ON A.away_team_api_id = T.team_api_id", con)
Partidos_Equipo
Partidos_Equipo['resultados_home'] = (Partidos_Equipo['home_team_goal']-Partidos_Equipo['away_team_goal'])

Partidos_Equipo['resultados_away'] = (Partidos_Equipo['away_team_goal']-Partidos_Equipo['home_team_goal'])



Partidos_Equipo
def set_puntos_h(row):

    if row["resultados_home"] < 0:

        return 0

    elif row["resultados_home"] == 0:

        return 1

    else:

        return 3



Partidos_Equipo = Partidos_Equipo.assign(home_puntos=Partidos_Equipo.apply(set_puntos_h, axis=1))



def set_puntos_a(row):

    if row["resultados_away"] < 0:

        return 0

    elif row["resultados_away"] == 0:

        return 1

    else:

        return 3



Partidos_Equipo = Partidos_Equipo.assign(away_puntos=Partidos_Equipo.apply(set_puntos_a, axis=1))

Partidos_Equipo
Partidos['dif_goles'] = pd.Series(Partidos['home_team_goal'] - Partidos['away_team_goal'], index = Partidos.index)



Partidos_dif_goles = Partidos.groupby(['dif_goles']).count()['id']

Partidos_dif_goles.plot(kind='bar')



dif_uno = Partidos_dif_goles[0]+Partidos_dif_goles[-1]+Partidos_dif_goles[1];

print (round(100*(dif_uno)/Partidos.shape[0], 2), " % de los partidos son decididos por un gol. ")



print (round(100*Partidos_dif_goles[0]/Partidos.shape[0], 2), " % son empates. " )
#Analizamos la diferencia puntos en casa respecto a los puntos fuera de casa.
Partidos_Equipo['dif_puntos'] = pd.Series(Partidos_Equipo['home_puntos'] - Partidos_Equipo['away_puntos'], index = Partidos_Equipo.index)

Puntos_home = Partidos_Equipo.groupby(['home_puntos']).count()['date']

Puntos_away= Partidos_Equipo.groupby(['away_puntos']).count()['date']
#Puntos en casa.

Puntos_home.plot(kind='bar')
#Analizamos los puntos fuera 

Puntos_away.plot(kind='bar')
Partidos_v1=Partidos.copy()





Partidos_v1.drop(Partidos_v1.columns[np.r_[11:115]], axis=1, inplace=True)
def ganadores(row):

    if row['home_team_goal'] > row['away_team_goal']:

        val = row["home_team_api_id"]

    elif row['home_team_goal'] < row['away_team_goal']:

        val = row["away_team_api_id"]

    else:

        val = None

    return val
Partidos_v1['GanadosID'] = Partidos_v1.apply(ganadores, axis=1)

Partidos_v1=Partidos_v1.dropna(thresh=12)

Partidos_v1['GanadosID'].astype(int)
Partidos_v1.info()
topGanadores = Partidos_v1[['match_api_id','GanadosID']].groupby(['GanadosID'])['match_api_id'].count().nlargest(35).reset_index(name='Nº_Partidos_Ganados')

topGanadores=pd.merge(topGanadores, Equipos,  how='left', left_on=['GanadosID'], right_on = ['team_api_id'])

topGanadores=topGanadores[['GanadosID','Nº_Partidos_Ganados','team_long_name']]

print(topGanadores)
topGanadores = Partidos_v1[['match_api_id','GanadosID']].groupby(['GanadosID'])['match_api_id'].count().nlargest(25978).reset_index(name='Nº_Partidos_Ganados')

topGanadores=pd.merge(topGanadores, Equipos,  how='left', left_on=['GanadosID'], right_on = ['team_api_id'])

topGanadores=topGanadores[['GanadosID','Nº_Partidos_Ganados','team_long_name']]

print(topGanadores.tail(35))