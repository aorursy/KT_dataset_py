import sqlite3

import pandas as pd

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from bokeh.plotting import figure, ColumnDataSource, show

from bokeh.models import HoverTool

from bokeh.io import output_notebook

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

output_notebook()



database = '../input/database.sqlite'

conn = sqlite3.connect(database)



# list all tables

query = "SELECT name as Tablas FROM sqlite_master WHERE type='table';"

pd.read_sql(query, conn)

query = "SELECT * FROM Player;"

a = pd.read_sql(query, conn)

a.head()



query = "SELECT * FROM Player_Attributes;"

a = pd.read_sql(query, conn)

a.head()

query = "SELECT max(a.gk_diving), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT max(a.gk_handling), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT max(a.gk_kicking), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT max(a.gk_positioning), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT max(a.gk_reflexes), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT a.gk_reflexes, a.date, p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_name='Iker Casillas';"

b = pd.read_sql(query, conn)

b.head()
import matplotlib.pyplot as plt

plt.plot(b['date'].str[:4], b['gk_reflexes'], marker='o', linestyle='-', color='r')

plt.show()
query = "SELECT max(media), id, c.media, c.player_name, c.date, c.gk_diving, c.gk_handling, c.gk_kicking, c.gk_positioning, c.gk_reflexes FROM (SELECT a.id,a.gk_diving, a.gk_handling, a.gk_kicking, a.gk_positioning, a.gk_reflexes, p.player_name,a.date, avg(a.gk_diving+a.gk_handling+a.gk_kicking+a.gk_positioning+a.gk_reflexes) as media FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id) as c;"

a = pd.read_sql(query, conn)

a.head()
query = "SELECT c.media, c.player_api_id, c.player_name, c.date, c.gk_diving, c.gk_handling, c.gk_kicking, c.gk_positioning, c.gk_reflexes FROM (SELECT a.id,p.player_api_id, a.gk_diving, a.gk_handling, a.gk_kicking, a.gk_positioning, a.gk_reflexes, p.player_name,a.date, avg(a.gk_diving+a.gk_handling+a.gk_kicking+a.gk_positioning+a.gk_reflexes) as media FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_name='Gianluigi Buffon') as c;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT c.media, c.player_name, c.date, c.gk_diving, c.gk_handling, c.gk_kicking, c.gk_positioning, c.gk_reflexes FROM (SELECT a.id,a.gk_diving, a.gk_handling, a.gk_kicking, a.gk_positioning, a.gk_reflexes, p.player_name,a.date, avg(a.gk_diving+a.gk_handling+a.gk_kicking+a.gk_positioning+a.gk_reflexes) as media FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_name='Iker Casillas') as c;"

b = pd.read_sql(query, conn)

b.head()
query ="SELECT p.player_api_id, a.id,a.gk_diving, a.gk_handling, a.gk_kicking, a.gk_positioning, a.gk_reflexes, p.player_name,a.date FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_api_id=30657;"

b = pd.read_sql(query, conn)

b.head()
import matplotlib.pyplot as plt

plt.plot(b['date'].str[:4], b['gk_reflexes'], marker='o', linestyle='-', color='r', label = "reflexes")

plt.plot(b['date'].str[:4], b['gk_diving'], marker='o', linestyle='-', color='b', label = "diving")

plt.plot(b['date'].str[:4], b['gk_kicking'], marker='o', linestyle='-', color='y',label = "kicking")

plt.plot(b['date'].str[:4], b['gk_positioning'], marker='o', linestyle='-', color='g',label = "positioning")

plt.plot(b['date'].str[:4], b['gk_handling'], marker='o', linestyle='-', color='c',label = "handling")

plt.title("Evolución capacidades Iker Casillas") 

plt.xlabel("Años") 

plt.ylabel("Puntuación")

plt.legend()

plt.show()
query ="SELECT p.player_api_id, a.id,a.gk_diving, a.gk_handling, a.gk_kicking, a.gk_positioning, a.gk_reflexes, p.player_name,a.date FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_api_id=30717;"

b = pd.read_sql(query, conn)

b.head()

import matplotlib.pyplot as plt

plt.plot(b['date'].str[:4], b['gk_reflexes'], marker='o', linestyle='-', color='r', label = "reflexes")

plt.plot(b['date'].str[:4], b['gk_diving'], marker='o', linestyle='-', color='b', label = "diving")

plt.plot(b['date'].str[:4], b['gk_kicking'], marker='o', linestyle='-', color='y',label = "kicking")

plt.plot(b['date'].str[:4], b['gk_positioning'], marker='o', linestyle='-', color='g',label = "positioning")

plt.plot(b['date'].str[:4], b['gk_handling'], marker='o', linestyle='-', color='c',label = "handling")

plt.title("Evolución capacidades Buffon") 

plt.xlabel("Años") 

plt.ylabel("Puntuación")

plt.legend()

plt.show()
query ="SELECT p.player_api_id, avg(a.gk_diving+a.gk_handling+a.gk_kicking+a.gk_positioning+a.gk_reflexes) as media, p.player_name,p.birthday FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id;"

players_data = pd.read_sql(query, conn)

players_data.head()
plt.plot(players_data['birthday'].str[:4], players_data['media'], marker='o', linestyle='-', color='r', label = "reflexes")



plt.title("Evaluación de las capacidades para ser un buen portero, para todos los jugadores") 

plt.xlabel("birthday") 

plt.ylabel("Puntuación")

plt.legend()

plt.show()
query ="SELECT p.player_api_id, avg(a.gk_diving+a.gk_handling+a.gk_kicking+a.gk_positioning+a.gk_reflexes) as media, p.player_name,a.overall_rating FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id;"

players_data = pd.read_sql(query, conn)

players_data.head()

plt.plot(players_data['overall_rating'], players_data['media'], marker='o', linestyle='-', color='r', label = "reflexes")



plt.title("Evaluación de las capacidades para ser un buen portero, para todos los jugadores") 

plt.xlabel("overall_rating") 

plt.ylabel("Puntuación")

plt.legend()

plt.show()
query = "SELECT * FROM Match;"

a = pd.read_sql(query, conn)

a.head()

query = "SELECT * FROM League;"

a = pd.read_sql(query, conn)

a.head()
query = "SELECT * FROM Country;"

a = pd.read_sql(query, conn)

a.head()
query = "SELECT * FROM Team;"

a = pd.read_sql(query, conn)

a.head()
query = "SELECT * FROM Team_Attributes;"

a = pd.read_sql(query, conn)

a.head()
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

    sql = "SELECT p.player_api_id, avg(a.gk_diving+a.gk_handling+a.gk_kicking+a.gk_positioning+a.gk_reflexes) as media, p.player_name,a.overall_rating FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id;"

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

    plt.title("Evaluación de las capacidades para ser un buen portero") 

    plt.xlabel("overall_rating") 

    plt.ylabel("Puntuación")



    subplot = fig.add_subplot(111)

    subplot.tick_params(axis='both', which='major', labelsize=22)

    midf  = subplot.scatter(midf_data["media"], midf_data["overall_rating"], marker='o', color="r", alpha = 0.5, s=50)

    defend = subplot.scatter(def_data["media"], def_data["overall_rating"], marker='o', color="g", alpha = 0.5, s=50)

    forw = subplot.scatter(forw_data["media"], forw_data["overall_rating"], marker='o', color="b", alpha = 0.5, s=50)

    gk  = subplot.scatter(gk_data["media"], gk_data["overall_rating"], marker='o', color="pink", alpha = 0.5, s=50)

    plt.xlabel('Cualidades de Portero', fontsize=30)

    plt.ylabel('overall_rating', fontsize=30)

    plt.legend((defend, forw, gk, midf),

           ('Defender', 'Forward', 'Goalkeeper', 'Midfielder'),

           scatterpoints=1,

           loc='down right',

           ncol=1,

           fontsize=20)

    plt.show()

plot_beautiful_scatter_position(players_data)
forw_data = players_data[players_data["position"] == "for"]

forw_data = forw_data[players_data["media"] >300]

forw_data.head()