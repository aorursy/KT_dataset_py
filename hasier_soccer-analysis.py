## Importar las librerías requeridas

import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



import warnings

warnings.filterwarnings('ignore')
## Obtención de datos

#Conectar a la base de datos

path = "C:/Users/Hasier/Downloads/Desktop/Formación/Master EOI/Módulo 17/Hasier/Actividad FINAL/"

#path = "../input/"

database = path + 'database.sqlite'

conn = sqlite3.connect(database)



#Listamos todas las tablas que la componen.

query = "SELECT name as TablasBD FROM sqlite_master WHERE type='table';"

table_soccer = pd.read_sql(query, conn)
#Visualizarmos las tablas existentes en la base de datos.

table_soccer
#Para visualizar la estructura de cada una de las tablas, he utilizado http://inloop.github.io/sqlite-viewer/

#Aun así, analizamos diferentes tablas con las que trabajaremos más adelante

#Obtenemos primero las tablas con las que vamos a trabajar

player = pd.read_sql("SELECT * FROM Player;", conn)

player_attributes = pd.read_sql("SELECT * FROM Player_Attributes;", conn)

team = pd.read_sql("SELECT * FROM Team;", conn)

team_attributes = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)

match = pd.read_sql("SELECT * FROM Match;", conn)

#Visualizamos por ejemlo la cabecera de la tabla de jugadores

player_attributes.head()
#Primero, voy a comprobar cuan relevante es el jugar como local o como visitante para el resultado del partido.



Ganalocal = (match[match.home_team_goal > match.away_team_goal]).id.count()

Empate = (match[match.home_team_goal == match.away_team_goal]).id.count()

Ganavisitante = (match[match.home_team_goal < match.away_team_goal]).id.count()

Total_Partidos = match.id.count()



print("% Victorias Locales:",Ganalocal*100/Total_Partidos)

print("% Empate:", Empate*100/Total_Partidos)

print("% Victorias Visitantes:", Ganavisitante*100/Total_Partidos)





values = [Ganalocal,Empate,Ganavisitante]

values = np.dot(values,100/Total_Partidos)



labels = ['Gana local','Empate','Gana visitante']

colors = ['lightgreen', 'yellow', 'red']



plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 

plt.axis('equal')

plt.show()

# En la liga española

liga_espanola = match[match.league_id==21518]



Ganalocal_Espana = (liga_espanola[liga_espanola.home_team_goal>liga_espanola.away_team_goal]).id.count()

Empate_Espana = (liga_espanola[liga_espanola.home_team_goal==liga_espanola.away_team_goal]).id.count()

Ganavisitante_Espana = (liga_espanola[liga_espanola.home_team_goal<liga_espanola.away_team_goal]).id.count()

Total_Partidos_Espana = liga_espanola.id.count()



values_liga_espanola = [Ganalocal_Espana,Empate_Espana,Ganavisitante_Espana]





print("% Victorias Locales:",Ganalocal_Espana*100/Total_Partidos_Espana)

print("% Empates:", Empate_Espana*100/Total_Partidos_Espana)

print("% Victorias Visitantes:", Ganavisitante_Espana*100/Total_Partidos_Espana)





values_liga_espanola = [Ganalocal_Espana,Empate_Espana,Ganavisitante_Espana]

values_liga_espanola = np.dot(values_liga_espanola,100/Total_Partidos_Espana)



labels = ['Gana local','Empate','Gana visitante']

colors = ['lightgreen', 'yellow', 'red']



plt.pie(values_liga_espanola, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 

plt.axis('equal')

plt.show()


liga_suiza = match[match.league_id==24558]



Ganalocal_Suiza = (liga_suiza[liga_suiza.home_team_goal>liga_suiza.away_team_goal]).id.count()

Empate_Suiza = (liga_suiza[liga_suiza.home_team_goal==liga_suiza.away_team_goal]).id.count()

Ganavisitante_Suiza = (liga_suiza[liga_suiza.home_team_goal<liga_suiza.away_team_goal]).id.count()

Total_Partidos_Suiza = liga_suiza.id.count()



values_liga_espanola = [Ganalocal_Suiza,Empate_Suiza,Ganavisitante_Suiza]





print("% Victorias Locales:",Ganalocal_Suiza*100/Total_Partidos_Suiza)

print("% Empates:", Empate_Suiza*100/Total_Partidos_Suiza)

print("% Victorias Visitantes:", Ganavisitante_Suiza*100/Total_Partidos_Suiza)





values_liga_suiza = [Ganalocal_Suiza,Empate_Suiza,Ganavisitante_Suiza]

values_liga_suiza = np.dot(values_liga_suiza,100/Total_Partidos_Suiza)



labels = ['Gana local','Empate','Gana visitante']

colors = ['lightgreen', 'yellow', 'red']



plt.pie(values_liga_suiza, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 

plt.axis('equal')

plt.show()


#Voy a analizar si esto se cumple para el Athletic Club de Bilbao, equipo históricamente y teóricamente fuerte en su campo.

#Para ello, cojo los registros en los que el Athletic juega como equipo local, para ver su % de victoria.



match_Bilbao = pd.read_sql("SELECT * FROM Match WHERE home_team_api_id = 8315;", conn)





Ganalocal_Bilbao = (match_Bilbao[match_Bilbao.home_team_goal>match_Bilbao.away_team_goal]).id.count()

Empate_Bilbao = (match_Bilbao[match_Bilbao.home_team_goal==match_Bilbao.away_team_goal]).id.count()

Ganavisitante_Bilbao = (match_Bilbao[match_Bilbao.home_team_goal<match_Bilbao.away_team_goal]).id.count()

Total_Partidos_Bilbao = match_Bilbao.id.count()



values_match_Bilbao = [Ganalocal_Bilbao,Empate_Bilbao,Ganavisitante_Bilbao]





print("% Victorias Locales:",Ganalocal_Bilbao*100/Total_Partidos_Bilbao)

print("% Empates:", Empate_Bilbao*100/Total_Partidos_Bilbao)

print("% Victorias Visitantes:", Ganavisitante_Bilbao*100/Total_Partidos_Bilbao)





values_liga_Bilbao = [Ganalocal_Bilbao,Empate_Bilbao,Ganavisitante_Bilbao]

values_liga_Bilbao = np.dot(values_liga_Bilbao,100/Total_Partidos_Bilbao)



labels = ['Gana Athletic','Empate','Gana visitante']

colors = ['lightgreen', 'yellow', 'red']



plt.pie(values_liga_Bilbao, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 

plt.axis('equal')

plt.show()
# Agrego una nueva columna



match_copy = match



match_copy['resultado'] = match_copy.home_team_goal - match_copy.away_team_goal



# Y selecciono los valores

match_subset = match_copy[['id','home_team_api_id','away_team_api_id','resultado']]



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# Añado la columna de media de atributos

team_attributes_copy = team_attributes



team_attributes_copy['Media_Equipo'] = team_attributes[['buildUpPlaySpeed','buildUpPlayDribbling','buildUpPlayPassing','chanceCreationShooting','chanceCreationShootingClass','defencePressure','defenceAggression','defenceTeamWidth']].mean(axis=1)



# Y ahora selecciono sólo los valores de interés

# Omito la fecha, ya que dificultaría obtener el resultado buscado, por tener medias diferentes por fecha. 

#Obviaremos ese detalle teniendo una sola media por equipo. Lo suyo sería filtrar por el campo fecha, únicamente en el año 2015, el más actual.

team_attributes_subset = team_attributes[['id','team_api_id','Media_Equipo']]

#Compruebo que funcion



print(match_subset.head())



print("\n")



# Hago un head y veo que ha ido bien todo

print(team_attributes_subset.head())

print("\n")

print("Máximo valor de resultado: ",match_subset.resultado.max())

print("Mìnimo valor de resultado: ",match_subset.resultado.min())

print("\n")

print("Máximo valor de resultado: ",match_subset.resultado.max())

print("Mìnimo valor de resultado: ",match_subset.resultado.min())

# Para el equipo local:

home_team_api_id_new = team_attributes_subset.rename(columns={'team_api_id': 'home_team_api_id'})

print(home_team_api_id_new.head())



print("\n")



# Y lo mismo para el equipo visitante

away_team_api_id_new = team_attributes_subset.rename(columns={'team_api_id': 'away_team_api_id'})

print(away_team_api_id_new.head())
# PARA EL EQUIPO LOCAL

# Hago el merge

train_soccer_1 = pd.merge(match_subset, home_team_api_id_new, on="home_team_api_id")

print(train_soccer_1.head())



print("\n")

# Y renombro como me interesa a mi

train_soccer_1b = train_soccer_1.rename(columns={'Media_Equipo': 'Media_Equipo_L'})

print(train_soccer_1b.head())
### REPITO PARA EL EQUIPO VISITANTE



train_soccer_2 = pd.merge(train_soccer_1b, away_team_api_id_new, on="away_team_api_id")

print(train_soccer_2.head())



# Y renombro como me interesa a mi

train_soccer_2b = train_soccer_2.rename(columns={'Media_Equipo': 'Media_Equipo_V'})

print(train_soccer_2b.head())

#Hago un subset:

train_soccer = train_soccer_2b[['Media_Equipo_L','Media_Equipo_V','resultado']]
# Quedaría:

print(train_soccer.head())



# Y hago una conversion de los results a string

train_soccer[['resultado']] = train_soccer[['resultado']].replace([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10], ['2','2','2','2','2','2','2','2','2','X','1', '1', '1','1', '1', '1','1', '1', '1','1'])

print("\n")

# Quedaría:

print(train_soccer.head())
# De esta forma podemos pasar a hacer nuestras primeras predicciones.

train_soccer['case1'] = train_soccer.Media_Equipo_L - train_soccer.Media_Equipo_V

train_soccer['case2'] = train_soccer.Media_Equipo_L - train_soccer.Media_Equipo_V

train_soccer['case3'] = train_soccer.Media_Equipo_L - train_soccer.Media_Equipo_V

print(train_soccer.head())
#**************************** CASO 1 **********************************************

# La primera y mas sencilla: el equipo que tenga atributos mas alto---> Ganador

def condition_case1(value):

    if value>0.0:

        return 1

    else:

        return 3

    

#**************************** CASO 2 **********************************************

# La segunda tiene en cuenta que si la diferencia es pequeña, habrá un empate

def condition_case2(value):

    if value>1:

        return 1

    elif value<-1:

        return 3

    else:

        return 2

    

#**************************** CASO 3 **********************************************

# La tercera tiene en cuenta que el factor casa. 

def condition_case3(value):

    if value>-1:

        return 1

    elif value<-3:

        return 3

    else:

        return 2
train_soccer['case1'] = train_soccer['case1'].apply(condition_case1)

print(train_soccer.head())



train_soccer['case2'] = train_soccer['case2'].apply(condition_case2)

print(train_soccer.head())



train_soccer['case3'] = train_soccer['case3'].apply(condition_case3)

print(train_soccer.head())
train_soccer[['case1']] = train_soccer[['case1']].replace([1, 2, 3], ['1', 'X', '2'])

train_soccer[['case2']] = train_soccer[['case2']].replace([1, 2, 3], ['1', 'X', '2'])

train_soccer[['case3']] = train_soccer[['case3']].replace([1, 2, 3], ['1', 'X', '2'])

print(train_soccer)

print("\n")
#**************************** CASO 1 **********************************************

# Numero de aciertos_

aciertos = np.sum(train_soccer.resultado == train_soccer.case1)

total = train_soccer.resultado.count()



print("Porcentaje de aciertos (case1): ", aciertos*100/total)



#**************************** CASO 2 **********************************************

# Numero de aciertos_

aciertos = np.sum(train_soccer.resultado == train_soccer.case2)

total = train_soccer.resultado.count()



print("Porcentaje de aciertos (case2): ", aciertos*100/total)



#**************************** CASO 3 **********************************************

# Numero de aciertos_

aciertos = np.sum(train_soccer.resultado == train_soccer.case3)

total = train_soccer.resultado.count()



print("Porcentaje de aciertos (case3): ", aciertos*100/total)