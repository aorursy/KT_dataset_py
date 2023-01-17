import pandas as pd

import sqlite3

import numpy as np

from IPython.display import display

import matplotlib.pyplot as plt

%matplotlib inline

conn = sqlite3.connect('../input/database.sqlite')
countries = pd.read_sql_query("SELECT * from Country", conn)

matches = pd.read_sql_query("SELECT * from Match", conn)

leagues = pd.read_sql_query("SELECT * from League", conn)

teams = pd.read_sql_query("SELECT * from Team", conn)

players = pd.read_sql_query("SELECT * from Player", conn)

player_attributes = pd.read_sql_query("SELECT * from Player_Attributes", conn)
#countries.head()

#matches.head()

#leagues.head()

#teams.head()

#players.head()

player_attributes.head()



#countries.info()

#matches.info()

#leagues.info()

#teams.info()

#players.info()

player_attributes.info()

query = "SELECT max(a.short_passing), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT max(a.free_kick_accuracy), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT max(a.long_passing), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT max(media), id, c.media, c.player_name, c.date, c.short_passing, c.free_kick_accuracy, c.long_passing FROM (SELECT a.id,a.short_passing, a.free_kick_accuracy, a.long_passing,  p.player_name,a.date, avg(a.short_passing+a.free_kick_accuracy+a.long_passing) as media FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id) as c;"

a = pd.read_sql(query, conn)

a.head()
query ="SELECT p.player_api_id,preferred_foot, avg(a.short_passing+a.free_kick_accuracy+a.long_passing) as media, p.player_name,p.birthday FROM player_attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id;"

players_data = pd.read_sql(query, conn)

players_data.head()
player_right = players_data[players_data.preferred_foot == 'right']

player_left =players_data[players_data.preferred_foot == 'left'] 

display('La precision media de un jugador diestro es de {:,.3f} puntos'.format(player_right.media.mean()))

display('Con una desviación típica {:,.3f} puntos'.format(player_right.media.std()))

display('Por su parte, la mediana es de {:,.3f} puntos'.format(player_right.media.median()))
display('La precision media de un jugador zurdo es de {:,.3f} puntos'.format(player_left.media.mean()))

display('Con una desviación típica {:,.3f} puntos'.format(player_left.media.std()))

display('Por su parte, la mediana es de {:,.3f} puntos'.format(player_left.media.median()))

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

conn = sqlite3.connect('../input/database.sqlite')

c = conn.cursor()

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

                    return "for"

                elif (mean_y > 5):

                    return "mid"

                elif (mean_y > 1):

                    return "def"

                elif (mean_y == 1.0):

                    return "gk"

    return None



         



with sqlite3.connect('../input/database.sqlite') as con:

    sql = "SELECT p.player_api_id, avg(a.short_passing+a.free_kick_accuracy+a.long_passing) as media, p.player_name,a.overall_rating FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id;"

    max_players_to_analyze = 1000

    players_data = pd.read_sql_query(sql, con)

    players_data = players_data.iloc[0:max_players_to_analyze]

    players_data["position"] = np.vectorize(get_position)(players_data["player_api_id"])

players_data.head()
def plot_beautiful_scatter_position(players_data):

    fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

    def_data = players_data[players_data["position"] == "def"]

    forw_data = players_data[players_data["position"] == "for"]

    gk_data = players_data[players_data["position"] == "gk"]

    midf_data =  players_data[players_data["position"] == "mid"]

    plt.title("relación calidad precision") 

    plt.xlabel("overall_rating") 

    plt.ylabel("Puntuación")



    subplot = fig.add_subplot(111)

    subplot.tick_params(axis='both', which='major', labelsize=22)

    midf  = subplot.scatter(midf_data["media"], midf_data["overall_rating"], marker='o', color="r", alpha = 0.5, s=50)

    defend = subplot.scatter(def_data["media"], def_data["overall_rating"], marker='o', color="g", alpha = 0.5, s=50)

    forw = subplot.scatter(forw_data["media"], forw_data["overall_rating"], marker='o', color="b", alpha = 0.5, s=50)

    gk  = subplot.scatter(gk_data["media"], gk_data["overall_rating"], marker='o', color="pink", alpha = 0.5, s=50)

    plt.xlabel('precision con el balon', fontsize=30)

    plt.ylabel('overall_rating', fontsize=30)

    plt.legend((defend, forw, gk, midf),

           ('Def', 'Forw', 'Goalkr', 'Midf'),

           scatterpoints=1,

           loc='down right',

           ncol=1,

           fontsize=10)

    plt.show()

plot_beautiful_scatter_position(players_data)
