# Siguiendo el proceso CRISP DM preparamos los datos para explorarlos y obtener insights.



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import folium

import numpy as np

from sklearn.preprocessing import LabelEncoder

import sqlite3

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline



data_base = sqlite3.connect('../input/database.sqlite')

pais = pd.read_sql_query("SELECT * from Country", data_base)

partidos = pd.read_sql_query("SELECT * from Match", data_base)

liga = pd.read_sql_query("SELECT * from League", data_base)

equipos = pd.read_sql_query("SELECT * from Team", data_base)

equipos_caracteristicas = pd.read_sql_query("SELECT * from Team_Attributes", data_base)

jugadores = pd.read_sql_query("SELECT * from Player", data_base)

jugadores_caracteristicas = pd.read_sql_query("SELECT * from Player_Attributes", data_base)



partidos.head()
liga.head()
equipos.head()
equipos_caracteristicas.head()
jugadores.head()
jugadores_caracteristicas.head()
equipos_caracteristicas.info()
query = "SELECT max(d.buildUpPlaySpeed), t.team_long_name FROM Team_Attributes d, team t where t.team_api_id=d.team_api_id ;"

c = pd.read_sql(query, data_base)

c.head()
query = "SELECT max(d.buildUpPlayPassing), t.team_long_name FROM Team_Attributes d, team t where t.team_api_id=d.team_api_id ;"

c = pd.read_sql(query, data_base)

c.head()
query = "SELECT max(d.chanceCreationShooting), t.team_long_name FROM Team_Attributes d, team t where t.team_api_id=d.team_api_id ;"

c = pd.read_sql(query, data_base)

c.head()
query = "SELECT max(media), id, e.media, e.team_long_name, e.buildUpPlaySpeed, e.buildUpPlayPassing, e.chanceCreationShooting FROM (SELECT d.id,d.buildUpPlaySpeed, d.buildUpPlayPassing, d.chanceCreationShooting,  t.team_long_name, avg(d.buildUpPlaySpeed + d.buildUpPlayPassing + d.chanceCreationShooting) as media FROM Team_Attributes d, team t where t.team_api_id=d.team_api_id group by t.team_api_id) as e;"

a = pd.read_sql(query, data_base)

a.head()

query ="SELECT t.team_api_id, defenceTeamWidth, avg(d.buildUpPlaySpeed + d.buildUpPlayPassing + d.chanceCreationShooting) as media, t.team_long_name FROM Team_Attributes d, team t where t.team_api_id=d.team_api_id group by t.team_api_id;"

equipos_data = pd.read_sql(query, data_base)

equipos_data.head()
equipos_defensa_ancha = equipos_data[equipos_data.defenceTeamWidth > 46]

equipos_defensa_angosta = equipos_data[equipos_data.defenceTeamWidth < 46]
from IPython.display import display
display('El favoritismo medio de un equipo de defensa ancha es de {:,.3f} puntos'.format(equipos_defensa_ancha.media.mean()))

display('Con una desviación típica {:,.3f} puntos'.format(equipos_defensa_ancha.media.std()))

display('Por su parte, la mediana es de {:,.3f} puntos'.format(equipos_defensa_ancha.media.median()))
display('El favoritismo medio de un equipo de defensa angosta es de {:,.3f} puntos'.format(equipos_defensa_angosta.media.mean()))

display('Con una desviación típica {:,.3f} puntos'.format(equipos_defensa_angosta.media.std()))

display('Por su parte, la mediana es de {:,.3f} puntos'.format(equipos_defensa_angosta.media.median()))
jugadores_caracteristicas.info()
import math

import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import datetime as dt

import datetime

import sqlalchemy

from numpy.random import random

from sqlalchemy import create_engine

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

data_base = sqlite3.connect('../input/database.sqlite')

c = data_base.cursor()

import datetime



def get_position(x):

    global c

    all_rating = c.execute("""SELECT overall_rating FROM Player_Attributes WHERE player_api_id = '%d' """ % (x)).fetchall()

    all_rating = np.array(all_rating,dtype=np.float)[:,0]

    rating = np.nanmean(all_rating)

    if (rating>1): 

        all_football_nums = reversed(range(1,12))

        for num in all_football_nums:

            all_y_coord = c.execute("""SELECT home_player_Y%d FROM Match WHERE home_player_%d = '%d'""" % (num,num,x)).fetchall()

            if len(all_y_coord) > 0:

                Y = np.array(all_y_coord,dtype=np.float)

                mean_y = np.nanmean(Y)

                if (mean_y >= 10.0):

                    return "delantero_excelente"

                elif (mean_y > 5):

                    return "delantero_completo"

                elif (mean_y > 1):

                    return "delantero_basico"

                elif (mean_y == 1.0):

                    return "no_delantero"

    return None



         



with sqlite3.connect('../input/database.sqlite') as con:

    sql = "SELECT p.player_api_id, avg(a.ball_control+a.acceleration+a.shot_power) as media, p.player_name,a.overall_rating FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id;"

    max_players_to_analyze = 1000

    players_data = pd.read_sql_query(sql, con)

    players_data = players_data.iloc[0:max_players_to_analyze]

    players_data["position"] = np.vectorize(get_position)(players_data["player_api_id"])

players_data.head()
def plot_beautiful_scatter_position(players_data):

    fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

    def_data = players_data[players_data["position"] == "delantero_basico"]

    forw_data = players_data[players_data["position"] == "delantero_excelente"]

    gk_data = players_data[players_data["position"] == "no_delantero"]

    midf_data =  players_data[players_data["position"] == "delantero_completo"]

    plt.title("relación calidad completitud como delantero") 

    plt.xlabel("overall_rating") 

    plt.ylabel("Puntuación")



    subplot = fig.add_subplot(111)

    subplot.tick_params(axis='both', which='major', labelsize=22)

    midf  = subplot.scatter(midf_data["media"], midf_data["overall_rating"], marker='o', color="r", alpha = 0.5, s=50)

    defend = subplot.scatter(def_data["media"], def_data["overall_rating"], marker='o', color="g", alpha = 0.5, s=50)

    forw = subplot.scatter(forw_data["media"], forw_data["overall_rating"], marker='o', color="b", alpha = 0.5, s=50)

    gk  = subplot.scatter(gk_data["media"], gk_data["overall_rating"], marker='o', color="pink", alpha = 0.5, s=50)

    plt.xlabel('completitud como delantero', fontsize=30)

    plt.ylabel('overall_rating', fontsize=30)

    plt.legend((defend, forw, gk, midf),

           ('delantero_basico', 'delantero_excelente', 'no_delantero', 'delantero_completo'),

           scatterpoints=1,

           loc='down right',

           ncol=1,

           fontsize=10)

    plt.show()

plot_beautiful_scatter_position(players_data)