# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import sqlite3

import numpy as np

from IPython.display import display

import matplotlib.pyplot as plt

%matplotlib inline

conn = sqlite3.connect('../input/database.sqlite')

leagues = pd.read_sql_query("SELECT * from League", conn)

teams = pd.read_sql_query("SELECT * from Team", conn)

countries = pd.read_sql_query("SELECT * from Country", conn)

player_attributes = pd.read_sql_query("SELECT * from Player_Attributes", conn)

matches = pd.read_sql_query("SELECT * from Match", conn)

players = pd.read_sql_query("SELECT * from Player", conn)
#Elegimos este codigo para ver los primeros datos de los diferentes jugadores.

player_attributes.head()
#Realizamos la función info para poder ver todos los tipos de caracteristicas de los jugadores.



player_attributes.info()
# Potential



query = "SELECT max(a.potential), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
# Dribbling



query = "SELECT max(a.dribbling), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
# Penalties



query = "SELECT max(a.penalties), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
# Aggression



query = "SELECT max(a.aggression), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT max(media), id, c.media, c.player_name, c.date, c.potential, c.dribbling, c.long_passing FROM (SELECT a.id,a.potential, a.dribbling, a.long_passing,  p.player_name,a.date, avg(a.potential+a.dribbling+a.long_passing) as media FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id) as c;"

a = pd.read_sql(query, conn)

a.head()
query ="SELECT p.player_api_id,attacking_work_rate, avg(a.short_passing+a.free_kick_accuracy+a.long_passing) as media, p.player_name,p.birthday FROM player_attributes a, player p where p.player_api_id=a.player_api_id group by p.player_api_id;"

players_data = pd.read_sql(query, conn)

players_data.head()
player_high = players_data[players_data.attacking_work_rate == 'high']

player_medium =players_data[players_data.attacking_work_rate == 'medium'] 
display('Precisión media de trabajo en ataque alta {:,.3f} puntos'.format(player_high.media.mean()))



display('Desviación típica {:,.3f} puntos'.format(player_high.media.std()))



display('Mediana {:,.3f} puntos'.format(player_high.media.median()))


display('Precisión media de trabajo en ataque media {:,.3f} puntos'.format(player_medium.media.mean()))



display('Desviación típica {:,.3f} puntos'.format(player_medium.media.std()))



display('Mediana {:,.3f} puntos'.format(player_medium.media.median()))

str(player_attributes)
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
query = "SELECT * FROM Player_Attributes;"

a = pd.read_sql(query, conn)

a.head()
query = "SELECT max(a.gk_positioning), p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ;"

b = pd.read_sql(query, conn)

b.head()
query = "SELECT a.gk_diving, a.date, p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_name='Gianluigi Buffon';"

b = pd.read_sql(query, conn)

b.head()
import matplotlib.pyplot as plt

plt.plot(b['date'].str[:4], b['gk_diving'], marker='o', linestyle='-', color='r')

plt.show()
query = "SELECT c.media, c.player_api_id, c.player_name, c.date, c.gk_diving, c.gk_handling, c.gk_kicking, c.gk_positioning, c.gk_reflexes FROM (SELECT a.id,p.player_api_id, a.gk_diving, a.gk_handling, a.gk_kicking, a.gk_positioning, a.gk_reflexes, p.player_name,a.date, avg(a.gk_diving+a.gk_handling+a.gk_kicking+a.gk_positioning+a.gk_reflexes) as media FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_name='Gianluigi Buffon') as c;"

b = pd.read_sql(query, conn)

b.head()
query ="SELECT p.player_api_id, a.id,a.gk_diving, a.gk_handling, a.gk_kicking, a.gk_positioning, a.gk_reflexes, p.player_name,a.date FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_api_id=30717;"

b = pd.read_sql(query, conn)

b.head()

import matplotlib.pyplot as plt

plt.plot(b['date'].str[:4], b['gk_reflexes'], marker='o', linestyle='-', color='r', label = "reflexes")

plt.plot(b['date'].str[:4], b['gk_diving'], marker='o', linestyle='-', color='b', label = "diving")

plt.plot(b['date'].str[:4], b['gk_kicking'], marker='o', linestyle='-', color='y',label = "kicking")

plt.plot(b['date'].str[:4], b['gk_positioning'], marker='o', linestyle='-', color='g',label = "positioning")

plt.plot(b['date'].str[:4], b['gk_handling'], marker='o', linestyle='-', color='c',label = "handling")

plt.title("Capacidades de Gianluigi Buffon") 

plt.xlabel("Años") 

plt.ylabel("Puntuación")

plt.legend()

plt.show()