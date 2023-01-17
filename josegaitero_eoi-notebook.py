import pandas as pd

import sqlite3

import numpy as np

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn

from scipy.stats import poisson,skellam

%matplotlib inline

conn = sqlite3.connect('../input/database.sqlite')

countries = pd.read_sql_query("SELECT * from Country", conn)

matches = pd.read_sql_query("SELECT * from Match", conn)

leagues = pd.read_sql_query("SELECT * from League", conn)

teams = pd.read_sql_query("SELECT * from Team", conn)



matches.head()
 

inicio = pd.read_sql_query("SELECT tabla1.team_long_name as Local,tabla2.team_long_name as Visitante,tabla3.home_team_goal,tabla3.away_team_goal  from team as tabla1, team as tabla2,Match as tabla3,country as tabla4  where tabla1.team_api_id=tabla3.home_team_api_id and tabla2.team_api_id=tabla3.away_team_api_id  and tabla4.id=tabla3.country_id and tabla4.name in('Spain') and tabla3.season='2008/2009' order by tabla3.date asc", conn)

inicio = inicio[:-10]

inicio.mean()



# Construimos la funcion de prediccion de Poisson

from scipy.stats import poisson,skellam

poisson_pred = np.column_stack([[poisson.pmf(i, inicio.mean()[j]) for i in range(8)] for j in range(2)])



# grafica de histograma

plt.hist(inicio[['home_team_goal', 'away_team_goal']].values, range(9), 

         alpha=0.7, label=['Casa', 'Fuera'],normed=True, color=["#FFA07A", "#20B2AA"])

# añadimos lineas para la distribucion de Poisson

pois1, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0],

                  linestyle='-', marker='o',label="Casa", color = '#CD5C5C')

pois2, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1],

                  linestyle='-', marker='o',label="Fuera", color = '#006400')

leg=plt.legend(loc='upper right', fontsize=13, ncol=2)

leg.set_title("Poisson           Actual        ", prop = {'size':'14', 'weight':'bold'})

##Añadimos etiquetas

plt.xticks([i-0.5 for i in range(1,9)],[i for i in range(9)])

plt.xlabel("Goles por Partido",size=13)

plt.ylabel("Proporcion de Partidos",size=13)

plt.title("Numero de Goles por Partido Liga Española 2008/2009",size=14,fontweight='bold')

plt.ylim([-0.004, 0.4])

plt.tight_layout()

plt.show()
# pProbabilidad de que un partido acabe en empate poniendo un 0, en la primera variable de la funcion

skellam.pmf(0.0,  inicio.mean()[0],  inicio.mean()[1])


# Probabilidad de que un equipo local gane por mas de dos goles

skellam.pmf(2,  inicio.mean()[0],  inicio.mean()[1])
##Ahora vamos a hacer un histograma de los datos con distribucion skellam y con una diferencia de goles entre

##8 a favor del equipo visitante y 8 a favor del equipo local



skellam_pred = [skellam.pmf(i,  inicio.mean()[0],  inicio.mean()[1]) for i in range(-8,8)]



plt.hist(inicio[['home_team_goal']].values - inicio[['away_team_goal']].values, range(-8,8), 

         alpha=0.7, label='Actual',normed=True)

plt.plot([i+0.5 for i in range(-8,8)], skellam_pred,

                  linestyle='-', marker='o',label="Skellam", color = '#CD5C5C')

plt.legend(loc='upper right', fontsize=13)

plt.xticks([i+0.5 for i in range(-8,8)],[i for i in range(-8,8)])

plt.xlabel("Goles Casa - Goles Fuera",size=13)

plt.ylabel("Partidos",size=13)

plt.title("Diferencia en goles marcados (Equipo Local vs Equipo Visitante)",size=14,fontweight='bold')

plt.ylim([-0.004, 0.26])

plt.tight_layout()

plt.show()
fig,(ax1,ax2) = plt.subplots(2, 1)





Barcelona_local = inicio[inicio['Local']=='FC Barcelona'][['home_team_goal']].apply(pd.value_counts,normalize=True)

Barcelona_local_pois = [poisson.pmf(i,np.sum(np.multiply(Barcelona_local.values.T,Barcelona_local.index.T),axis=1)[0]) for i in range(8)]

Recre_local = inicio[inicio['Local']=='RC Recreativo'][['home_team_goal']].apply(pd.value_counts,normalize=True)

Recreativo_local_pois = [poisson.pmf(i,np.sum(np.multiply(Recre_local.values.T,Recre_local.index.T),axis=1)[0]) for i in range(8)]

Barcelona_fuera = inicio[inicio['Visitante']=='FC Barcelona'][['away_team_goal']].apply(pd.value_counts,normalize=True)

Barcelona_fuera_pois = [poisson.pmf(i,np.sum(np.multiply(Barcelona_fuera.values.T,Barcelona_fuera.index.T),axis=1)[0]) for i in range(8)]

Recre_fuera = inicio[inicio['Visitante']=='RC Recreativo'][['away_team_goal']].apply(pd.value_counts,normalize=True)

Recreativo_fuera_pois = [poisson.pmf(i,np.sum(np.multiply(Recre_fuera.values.T,Recre_fuera.index.T),axis=1)[0]) for i in range(8)]



ax1.bar(Barcelona_local.index-0.4,Barcelona_local.values,width=0.4,color="#034694",label="Barcelona")

ax1.bar(Recre_local.index,Recre_local.values,width=0.4,color="#EB172B",label="Recreativo")

pois1, = ax1.plot([i for i in range(8)], Barcelona_local_pois,

                  linestyle='-', marker='o',label="Barcelona", color = "#0a7bff")

pois1, = ax1.plot([i for i in range(8)], Recreativo_local_pois,

                  linestyle='-', marker='o',label="Recreativo", color = "#ff7c89")

leg=ax1.legend(loc='upper right', fontsize=12, ncol=2)

leg.set_title("Poisson                 Actual                ", prop = {'size':'14', 'weight':'bold'})

ax1.set_xlim([-0.5,7.5])

ax1.set_ylim([-0.01,0.65])

ax1.set_xticklabels([])

ax1.text(7.65, 0.585, '                Local               ', rotation=-90,

        bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})

ax2.text(7.65, 0.585, '                 Visitante                ', rotation=-90,

        bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})



ax2.bar(Barcelona_fuera.index-0.4,Barcelona_fuera.values,width=0.4,color="#034694",label="Barcelona")

ax2.bar(Recre_fuera.index,Recre_fuera.values,width=0.4,color="#EB172B",label="Recreativo")

pois1, = ax2.plot([i for i in range(8)], Barcelona_fuera_pois,

                  linestyle='-', marker='o',label="Barcelona", color = "#0a7bff")

pois1, = ax2.plot([i for i in range(8)], Recreativo_fuera_pois,

                  linestyle='-', marker='o',label="Recreativo", color = "#ff7c89")

ax2.set_xlim([-0.5,7.5])

ax2.set_ylim([-0.01,0.65])

ax1.set_title("Numero de Goles por Partido Liga Española 2008-2009",size=14,fontweight='bold')

ax2.set_xlabel("Goles por Partido",size=13)

ax2.text(-1.15, 0.9, 'Partidos', rotation=90, size=13)

plt.tight_layout()

plt.show()
##Lo primero que vamos a hacer es importarnos las librerias necesarias  para crear el modelo de regresion

import statsmodels.api as sm

import statsmodels.formula.api as smf

##Vamos a hacer un modelo concatenando los datos de  la variable de equipo local

##que nos dara la influencia de los goles marcados de los equipos locales, la variable equipo visitante los goles marcados por los equipos

##visitantes, y una variable de probabilidad de equipo local que sera 1 cuando los goles sean marcados por 

##eauipos locales y 0 cuando los goles sean marcados por los visitantes. Vamos a hacer dos clasificaciones entendiendo

##cuando los esuipos locales marcan llamando a la variable eauipo anotador y a los visitantes equipo oponente, y 

##cuando marcan los eauipos visitantes llamado eauipo anotador y a los locales eauipo oponente.

##Con esto el modelo nos va a dar varias predicciones de como un eauipo le cuesta marcar un go a otro equipo

##si esta variable del modelo es psitiva será que le cuesta poco hacer gol, cuando es negativa es que le cuesta mucho hacer gol

##En cambio para las variables de los oponentes será cuanto es dificil hacerle un gol a un eauipo, si es negativa el

##coeficieciente de la regresion significa que le va a costar mucho hacerle un gol y si es positiva es que es mas fcail hacerle un gol

##la variable local nos va a decir si los esuipos locales hacen mas goles que los visitantes si es positiva su coeficiente

modelo = pd.concat([inicio[['Local','Visitante','home_team_goal']].assign(local=1).rename(

            columns={'Local':'equipo', 'Visitante':'oponente','home_team_goal':'goles'}),

           inicio[['Visitante','Local','away_team_goal']].assign(local=0).rename(

            columns={'Visitante':'equipo', 'Local':'oponente','away_team_goal':'goles'})])



poisson_model = smf.glm(formula="goles ~ local + equipo + oponente", data=modelo, 

                        family=sm.families.Poisson()).fit()

poisson_model.summary()
poisson_model.predict(pd.DataFrame(data={'equipo': 'FC Barcelona', 'oponente': 'RC Recreativo',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'RC Recreativo', 'oponente': 'FC Barcelona',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'RC Recreativo', 'oponente': 'FC Barcelona',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'FC Barcelona', 'oponente': 'RC Recreativo',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'Atlético Madrid', 'oponente': 'UD Almería',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'UD Almería', 'oponente': 'Atlético Madrid',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'Valencia CF', 'oponente': 'Athletic Club de Bilbao',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'Athletic Club de Bilbao', 'oponente': 'Valencia CF',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'RC Deportivo de La Coruña', 'oponente': 'FC Barcelona',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'oponente': 'RC Deportivo de La Coruña', 'equipo': 'FC Barcelona',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'RCD Mallorca', 'oponente': 'Villarreal CF',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'oponente': 'RCD Mallorca', 'equipo': 'Villarreal CF',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'RCD Espanyol', 'oponente': 'Málaga CF',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'oponente': 'RCD Espanyol', 'equipo': 'Málaga CF',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'CD Numancia', 'oponente': 'Sevilla FC',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'oponente': 'CD Numancia', 'equipo': 'Sevilla FC',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'Racing Santander', 'oponente': 'Getafe CF',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'oponente': 'Racing Santander', 'equipo': 'Getafe CF',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'CA Osasuna', 'oponente': 'Real Madrid CF',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'oponente': 'CA Osasuna', 'equipo': 'Real Madrid CF',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'Real Sporting de Gijón', 'oponente': 'RC Recreativo',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'oponente': 'Real Sporting de Gijón', 'equipo': 'RC Recreativo',

                                       'local':0},index=[1]))
poisson_model.predict(pd.DataFrame(data={'equipo': 'Real Betis Balompié', 'oponente': 'Real Valladolid',

                                       'local':1},index=[1]))
poisson_model.predict(pd.DataFrame(data={'oponente': 'Real Betis Balompié', 'equipo': 'Real Valladolid',

                                       'local':0},index=[1]))