 %matplotlib inline



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sqlite3

from numpy import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Siguiendo el paso 2 de CRISP-DM, efectuamos un conteo de las filas que contienen al menos un NULL

# en la base de datos. El número de filas por tabla viene dado en la descripción de la base de

# datos, por lo que no hay que calcularlo.

with sqlite3.connect('../input/database.sqlite') as con:

    player_attributes_null_amount = pd.read_sql_query("SELECT COUNT (*) from Player_Attributes WHERE id IS NULL OR player_fifa_api_id IS NULL OR player_api_id IS NULL OR date IS NULL OR overall_rating IS NULL OR potential IS NULL OR preferred_foot IS NULL OR attacking_work_rate IS NULL OR defensive_work_rate IS NULL OR crossing IS NULL OR finishing IS NULL OR heading_accuracy IS NULL OR short_passing IS NULL OR volleys IS NULL OR dribbling IS NULL OR curve IS NULL OR free_kick_accuracy IS NULL OR long_passing IS NULL OR ball_control IS NULL OR acceleration IS NULL OR sprint_speed IS NULL OR agility IS NULL OR reactions IS NULL OR balance IS NULL OR shot_power IS NULL OR jumping IS NULL OR stamina IS NULL OR strength IS NULL OR long_shots IS NULL OR aggression IS NULL OR interceptions IS NULL OR positioning IS NULL OR vision IS NULL OR penalties IS NULL OR marking IS NULL OR standing_tackle IS NULL OR sliding_tackle IS NULL OR gk_diving IS NULL OR gk_handling IS NULL OR gk_kicking IS NULL OR gk_positioning IS NULL OR gk_reflexes IS NULL", con)

    player_null_amount = pd.read_sql_query("SELECT COUNT (*) from Player WHERE id IS NULL OR player_api_id IS NULL OR player_name IS NULL OR player_fifa_api_id IS NULL OR birthday IS NULL OR height IS NULL OR weight IS NULL", con)

    match_null_amount = pd.read_sql_query("SELECT COUNT (*) from Match WHERE id IS NULL OR country_id IS NULL OR league_id IS NULL OR season IS NULL OR stage IS NULL OR date IS NULL OR match_api_id IS NULL OR home_team_api_id IS NULL OR away_team_api_id IS NULL OR home_team_goal IS NULL OR away_team_goal IS NULL OR home_player_X1 IS NULL OR home_player_X2 IS NULL OR home_player_X3 IS NULL OR home_player_X4 IS NULL OR home_player_X5 IS NULL OR home_player_X6 IS NULL OR home_player_X7 IS NULL OR home_player_X8 IS NULL OR home_player_X9 IS NULL OR home_player_X10 IS NULL OR home_player_X11 IS NULL OR away_player_X1 IS NULL OR away_player_X2 IS NULL OR away_player_X3 IS NULL OR away_player_X4 IS NULL OR away_player_X5 IS NULL OR away_player_X6 IS NULL OR away_player_X7 IS NULL OR away_player_X8 IS NULL OR away_player_X9 IS NULL OR away_player_X10 IS NULL OR away_player_X11 IS NULL OR home_player_Y1 IS NULL OR home_player_Y2 IS NULL OR home_player_Y3 IS NULL OR home_player_Y4 IS NULL OR home_player_Y5 IS NULL OR home_player_Y6 IS NULL OR home_player_Y7 IS NULL OR home_player_Y8 IS NULL OR home_player_Y9 IS NULL OR home_player_Y10 IS NULL OR home_player_Y11 IS NULL OR away_player_Y1 IS NULL OR away_player_Y2 IS NULL OR away_player_Y3 IS NULL OR away_player_Y4 IS NULL OR away_player_Y5 IS NULL OR away_player_Y6 IS NULL OR away_player_Y7 IS NULL OR away_player_Y8 IS NULL OR away_player_Y9 IS NULL OR away_player_Y10 IS NULL OR away_player_Y11 IS NULL OR home_player_1 IS NULL OR home_player_2 IS NULL OR home_player_3 IS NULL OR home_player_4 IS NULL OR home_player_5 IS NULL OR home_player_6 IS NULL OR home_player_7 IS NULL OR home_player_8 IS NULL OR home_player_9 IS NULL OR home_player_10 IS NULL OR home_player_11 IS NULL OR away_player_1 IS NULL OR away_player_2 IS NULL OR away_player_3 IS NULL OR away_player_4 IS NULL OR away_player_5 IS NULL OR away_player_6 IS NULL OR away_player_7 IS NULL OR away_player_8 IS NULL OR away_player_9 IS NULL OR away_player_10 IS NULL OR away_player_11 IS NULL OR goal IS NULL OR shoton IS NULL OR shotoff IS NULL OR foulcommit IS NULL OR card IS NULL OR cross IS NULL OR corner IS NULL OR possession IS NULL OR B365H IS NULL OR B365D IS NULL OR B365A IS NULL OR BWH IS NULL OR BWD IS NULL OR BWA IS NULL OR IWH IS NULL OR IWD IS NULL OR IWA IS NULL OR LBH IS NULL OR LBD IS NULL OR LBA IS NULL OR PSH IS NULL OR PSD IS NULL OR PSA IS NULL OR WHH IS NULL OR WHD IS NULL OR WHA IS NULL OR SJH IS NULL OR SJD IS NULL OR SJA IS NULL OR VCH IS NULL OR VCD IS NULL OR VCA IS NULL OR GBH IS NULL OR GBD IS NULL OR GBA IS NULL OR BSH IS NULL OR BSD IS NULL OR BSA IS NULL", con)

    league_null_amount = pd.read_sql_query("SELECT COUNT (*) from League WHERE id IS NULL OR country_id IS NULL OR name IS NULL", con)

    country_null_amount = pd.read_sql_query("SELECT COUNT (*) from Country WHERE id IS NULL or name IS NULL", con)

    team_null_amount = pd.read_sql_query("SELECT COUNT (*) from Team WHERE id IS NULL OR team_api_id IS NULL OR team_fifa_api_id IS NULL OR team_long_name IS NULL OR team_short_name IS NULL", con)

    team_attributes_null_amount = pd.read_sql_query("SELECT COUNT (*) from Team_Attributes WHERE id IS NULL OR team_fifa_api_id IS NULL OR team_api_id IS NULL OR date IS NULL OR buildUpPlaySpeed IS NULL OR buildUpPlaySpeedClass IS NULL OR buildUpPlayDribbling IS NULL OR buildUpPlayDribblingClass IS NULL OR buildUpPlayPassing IS NULL OR buildUpPlayPassingClass IS NULL OR buildUpPlayPositioningClass IS NULL OR chanceCreationPassing IS NULL OR chanceCreationPassingClass IS NULL OR chanceCreationCrossing IS NULL OR chanceCreationCrossingClass IS NULL OR chanceCreationShooting IS NULL OR chanceCreationShootingClass IS NULL OR chanceCreationPositioningClass IS NULL OR defencePressure IS NULL OR defencePressureClass IS NULL OR defenceAggression IS NULL OR defenceAggressionClass IS NULL OR defenceTeamWidth IS NULL OR defenceTeamWidthClass IS NULL OR defenceDefenderLineClass IS NULL", con)

con.close()
#print ("Número de filas (de 183978) con algún NULL en la tabla Player_Attributes:")    

#print (player_attributes_null_amount.head())

#print ("*************************************************************************")

#print ("Número de filas (de 11060) con algún NULL en la tabla Player:")    

#print (player_null_amount.head())

#print ("*************************************************************************")

#print ("Número de filas (de 25979) con algún NULL en la tabla Match:")    

#print (match_null_amount.head())

#print ("*************************************************************************")

#print ("Número de filas (de 11) con algún NULL en la tabla League:")    

#print (league_null_amount.head())

#print ("*************************************************************************")

#print ("Número de filas (de 11) con algún NULL en la tabla Countries:")    

#print (country_null_amount.head())

#print ("*************************************************************************")

#print ("Número de filas (de 299) con algún NULL en la tabla Team:")    

#print (team_null_amount.head())

#print ("*************************************************************************")

#print ("Número de filas (de 1458) con algún NULL en la tabla Team_Attributes:")    

#print (team_attributes_null_amount.head())

#print ("*************************************************************************")
with sqlite3.connect('../input/database.sqlite') as con:

    player_attributes_all = pd.read_sql_query("SELECT * from Player_Attributes", con)

    player_all = pd.read_sql_query("SELECT * from Player", con)

    match_all = pd.read_sql_query("SELECT * from Match", con)

    league_all = pd.read_sql_query("SELECT * from League", con)

    country_all = pd.read_sql_query("SELECT * from Country", con)

    team_all = pd.read_sql_query("SELECT * from Team", con)

    team_attributes_all = pd.read_sql_query("SELECT * from Team_Attributes", con)

con.close()



#print ("Aspecto del contenido de la tabla Player_Attributes:\n")

#print (player_attributes_all.head())

#print ("*************************************************************************")

#print ("Aspecto del contenido de la tabla Player:\n")    

#print (player_all.head())

#print ("*************************************************************************")

#print ("Aspecto del contenido de la tabla Match:\n")    

#print (match_all.head())

#print ("*************************************************************************")

#print ("Aspecto del contenido de la tabla League:\n")    

#print (league_all.head())

#print ("*************************************************************************")

#print ("Aspecto del contenido de la tabla Countries:\n")    

#print (country_all.head())

#print ("*************************************************************************")

#print ("Aspecto del contenido de la tabla Team:\n")    

#print (team_all.head())

#print ("*************************************************************************")

#print ("Aspecto del contenido de la tabla Team_Attributes:\n")    

#print (team_attributes_all.head())

#print ("*************************************************************************")

# Para la totalidad de las ligas

vistoriasLocal = (match_all[match_all.home_team_goal>match_all.away_team_goal]).id.count()

empates = (match_all[match_all.home_team_goal==match_all.away_team_goal]).id.count()

vistoriasVisitante = (match_all[match_all.home_team_goal<match_all.away_team_goal]).id.count()

totalPartidos = match_all.id.count()



print("% Victorias Locales:",vistoriasLocal*100/totalPartidos)

print("% Empates:", empates*100/totalPartidos)

print("% Victorias Visitantes:", vistoriasVisitante*100/totalPartidos)





values = [vistoriasLocal,empates,vistoriasVisitante]

values = np.dot(values,100/totalPartidos)



labels = ['vistoriasLocal','empates','vistoriasVisitante']

colors = ['gold', 'yellowgreen', 'lightcoral']



plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 

plt.axis('equal')

plt.title('% de victorias en función de jugar en casa o fuera')

plt.show()
# En la liga española

liga_espanola = match_all[match_all.league_id==21518]



vistoriasLocal_liga_espanola = (liga_espanola[liga_espanola.home_team_goal>liga_espanola.away_team_goal]).id.count()

empates_liga_espanola = (liga_espanola[liga_espanola.home_team_goal==liga_espanola.away_team_goal]).id.count()

vistoriasVisitante_liga_espanola = (liga_espanola[liga_espanola.home_team_goal<liga_espanola.away_team_goal]).id.count()

totalPartidos_liga_espanola = liga_espanola.id.count()



values_liga_espanola = [vistoriasLocal_liga_espanola,empates_liga_espanola,vistoriasVisitante_liga_espanola]





print("% Victorias Locales:",vistoriasLocal_liga_espanola*100/totalPartidos_liga_espanola)

print("% Empates:", empates_liga_espanola*100/totalPartidos_liga_espanola)

print("% Victorias Visitantes:", vistoriasVisitante_liga_espanola*100/totalPartidos_liga_espanola)





values_liga_espanola = [vistoriasLocal_liga_espanola,empates_liga_espanola,vistoriasVisitante_liga_espanola]

values_liga_espanola = np.dot(values_liga_espanola,100/totalPartidos_liga_espanola)



labels = ['vistoriasLocal','empates','vistoriasVisitante']

colors = ['gold', 'yellowgreen', 'lightcoral']



plt.pie(values_liga_espanola, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 

plt.title('Liga española: % de victorias en función de jugar en casa o fuera')

plt.axis('equal')

plt.show()
# En francia

liga_francesa = match_all[match_all.league_id==4769]



vistoriasLocal_liga_francesa = (liga_francesa[liga_francesa.home_team_goal>liga_francesa.away_team_goal]).id.count()

empates_liga_francesa = (liga_francesa[liga_francesa.home_team_goal==liga_francesa.away_team_goal]).id.count()

vistoriasVisitante_liga_francesa = (liga_francesa[liga_francesa.home_team_goal<liga_francesa.away_team_goal]).id.count()

totalPartidos_liga_francesa = liga_francesa.id.count()



values_liga_francesa = [vistoriasLocal_liga_francesa,empates_liga_francesa,vistoriasVisitante_liga_francesa]





print("% Victorias Locales:",vistoriasLocal_liga_francesa*100/totalPartidos_liga_francesa)

print("% Empates:", empates_liga_francesa*100/totalPartidos_liga_francesa)

print("% Victorias Visitantes:", vistoriasVisitante_liga_francesa*100/totalPartidos_liga_francesa)





values_liga_francesa = [vistoriasLocal_liga_francesa,empates_liga_francesa,vistoriasVisitante_liga_francesa]

values_liga_francesa = np.dot(values_liga_francesa,100/totalPartidos_liga_francesa)



labels = ['vistoriasLocal','empates','vistoriasVisitante']

colors = ['gold', 'yellowgreen', 'lightcoral']



plt.pie(values_liga_francesa, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 

plt.title('Liga francesa: % de victorias en función de jugar en casa o fuera')

plt.axis('equal')

plt.show()

# En inglaterra

liga_inglesa = match_all[match_all.league_id==1729]



vistoriasLocal_liga_inglesa = (liga_inglesa[liga_inglesa.home_team_goal>liga_inglesa.away_team_goal]).id.count()

empates_liga_inglesa = (liga_inglesa[liga_inglesa.home_team_goal==liga_inglesa.away_team_goal]).id.count()

vistoriasVisitante_liga_inglesa = (liga_inglesa[liga_inglesa.home_team_goal<liga_inglesa.away_team_goal]).id.count()

totalPartidos_liga_inglesa = liga_inglesa.id.count()



values_liga_inglesa = [vistoriasLocal_liga_inglesa,empates_liga_inglesa,vistoriasVisitante_liga_inglesa]





print("% Victorias Locales:",vistoriasLocal_liga_inglesa*100/totalPartidos_liga_inglesa)

print("% Empates:", empates_liga_inglesa*100/totalPartidos_liga_inglesa)

print("% Victorias Visitantes:", vistoriasVisitante_liga_inglesa*100/totalPartidos_liga_inglesa)





values_liga_inglesa = [vistoriasLocal_liga_inglesa,empates_liga_inglesa,vistoriasVisitante_liga_inglesa]

values_liga_inglesa = np.dot(values_liga_inglesa,100/totalPartidos_liga_inglesa)



labels = ['vistoriasLocal','empates','vistoriasVisitante']

colors = ['gold', 'yellowgreen', 'lightcoral']



plt.pie(values_liga_inglesa, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 

plt.title('Liga inglesa: % de victorias en función de jugar en casa o fuera')

plt.axis('equal')

plt.show()

# Hago la copia

match_all_copy = match_all



# Agrego una nueva columna

match_all_copy['result'] = match_all_copy.home_team_goal - match_all_copy.away_team_goal



# Y selecciono los valores

match_all_subset = match_all_copy[['id','date','home_team_api_id','away_team_api_id','result']]





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# Hago la copia

team_atributes_all_copy = team_attributes_all



# Creo la nueva columna

team_atributes_all_copy['atributesMean'] = team_attributes_all[['buildUpPlaySpeed','buildUpPlayDribbling','buildUpPlayPassing','chanceCreationShooting','chanceCreationShootingClass','defencePressure','defenceAggression','defenceTeamWidth']].mean(axis=1)



# Y ahora selecciono sólo los valores de interés

team_atributes_all_subset = team_attributes_all[['id','team_fifa_api_id','team_api_id','date','atributesMean']]
# Hago un head y veo que ha ido bien todo

print(match_all_subset.head())



print("\n")



# Hago un head y veo que ha ido bien todo

print(team_atributes_all_subset.head())



print("Máximo valor de result: ",match_all_subset.result.max())

print("Mìnimo valor de result: ",match_all_subset.result.min())
# Calculo el numero total de equipos unicos:

unique_teams = np.size(team_atributes_all_subset.team_api_id.unique())



# Comienzo a crear el nuevo dataset

team_atributes_unique_new = pd.DataFrame(np.nan, index=np.arange(unique_teams), columns=['id','team_api_id','team_atributes_unique'])

team_atributes_unique_new.head()



# El id lo escribo como secuencia de consecutivos

team_atributes_unique_new['id'] = np.arange(unique_teams)



# Y los team_api_id son los que tengo ya calculados

team_atributes_unique_new['team_api_id'] = team_atributes_all_subset.team_api_id.unique()



# Compruebo con el head

team_atributes_unique_new.head()

# Ahora necesito asignar valores a team_atributes_unique

# Recorro los ids de los equipos y calculo el promedio de los atributos, y lo asigno a esa fila de ese equipo

for i in range(unique_teams):

    #print(i)

    teamId = (team_atributes_unique_new.iloc[i][1])

    #print(teamId)

    meanTeamAtributes = team_atributes_all_subset[team_atributes_all_subset.team_api_id==teamId].atributesMean.mean()

    team_atributes_unique_new.set_value(i,'team_atributes_unique',meanTeamAtributes)
team_atributes_unique_new.head()



# Y consigo lo que quería: un valor de lo bueno o malo que es un equipo en todo momento temporal
# Para el equipo local, aqui hago una pequeña trampa y renombro para adecuarlo al nombre de cara al merge

home_team_api_id_new = team_atributes_unique_new.rename(columns={'team_api_id': 'home_team_api_id'})

print(home_team_api_id_new.head())



print("\n")



# Y lo mismo para el equipo visitante

away_team_api_id_new = team_atributes_unique_new.rename(columns={'team_api_id': 'away_team_api_id'})

print(away_team_api_id_new.head())
# PARA EL EQUIPO LOCAL

# Hago el merge

train_soccer_1 = pd.merge(match_all_subset, home_team_api_id_new, on="home_team_api_id")

print(train_soccer_1.head())



print("\n")

# Y renombro como me interesa a mi

train_soccer_1b = train_soccer_1.rename(columns={'team_atributes_unique': 'home_team_atributes'})

print(train_soccer_1b.head())
### REPITO PARA EL EQUIPO VISITANTE

# Aqui hago una pequeña trampa y renombro para adecuarlo al nombre de cara al merge

train_soccer_2 = pd.merge(train_soccer_1b, away_team_api_id_new, on="away_team_api_id")

train_soccer_2.head()



# Y renombro como me interesa a mi

train_soccer_2b = train_soccer_2.rename(columns={'team_atributes_unique': 'away_team_atributes'})

print(train_soccer_2b.head())



# Y ya casi lo tengo. Hago un subset:

train_soccer = train_soccer_2b[['date','home_team_atributes','away_team_atributes','result']]

# Quedaría:

print(train_soccer.head())



# Y hago una conversion de los results a string

train_soccer[['result']] = train_soccer[['result']].replace([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10], ['Away','Away','Away','Away','Away','Away','Away','Away','Away','Draw','Home', 'Home', 'Home','Home', 'Home', 'Home','Home', 'Home', 'Home','Home'])

print("\n")

# Quedaría:

print(train_soccer.head())
# De esta forma podemos pasar a hacer nuestras primeras predicciones.

train_soccer['case1'] = train_soccer.home_team_atributes - train_soccer.away_team_atributes

train_soccer['case2'] = train_soccer.home_team_atributes - train_soccer.away_team_atributes

train_soccer['case3'] = train_soccer.home_team_atributes - train_soccer.away_team_atributes

train_soccer.tail()
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

train_soccer.head()



train_soccer['case2'] = train_soccer['case2'].apply(condition_case2)

train_soccer.head()



train_soccer['case3'] = train_soccer['case3'].apply(condition_case3)

train_soccer.head()
train_soccer[['case1']] = train_soccer[['case1']].replace([1, 2, 3], ['Home', 'Draw', 'Away'])

train_soccer[['case2']] = train_soccer[['case2']].replace([1, 2, 3], ['Home', 'Draw', 'Away'])

train_soccer[['case3']] = train_soccer[['case3']].replace([1, 2, 3], ['Home', 'Draw', 'Away'])

print(train_soccer.head())

print("\n")
#**************************** CASO 1 **********************************************

# Numero de aciertos_

aciertos = np.sum(train_soccer.result == train_soccer.case1)

total = train_soccer.result.count()



print("Porcentaje de aciertos (case1): ", aciertos*100/total)



#**************************** CASO 2 **********************************************

# Numero de aciertos_

aciertos = np.sum(train_soccer.result == train_soccer.case2)

total = train_soccer.result.count()



print("Porcentaje de aciertos (case2): ", aciertos*100/total)



#**************************** CASO 3 **********************************************

# Numero de aciertos_

aciertos = np.sum(train_soccer.result == train_soccer.case3)

total = train_soccer.result.count()



print("Porcentaje de aciertos (case3): ", aciertos*100/total)
from sklearn.cluster import KMeans

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split
X = train_soccer[['home_team_atributes','away_team_atributes']]

y = train_soccer[['result']]



# Y ahora divido el set de datos en dos grupos (70% de train y 30% de validacion)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train.head(3))

print("\n", y_train.head(3))
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)

kmeans.labels_



predictions = kmeans.predict(X_test)

#print(predictions)



y_test2 =  y_test.replace(['Home','Draw','Away'],[1,2,3])
# Score

# Numero de aciertos_

aciertos = np.sum(y_test2.result == predictions)

total = train_soccer.result.count()



print("Porcentaje de aciertos ", aciertos*100/total)