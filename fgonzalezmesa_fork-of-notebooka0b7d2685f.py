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
# los mejores jugadores 

query = "SELECT max(overall_rating),player_name,player_api_id FROM Player_Attributes a INNER JOIN (SELECT player_name, player_api_id AS p_id FROM Player) b ON a.player_api_id = b.p_id where overall_rating > 91 group by player_api_id order by overall_rating desc;"



pp=pd.read_sql(query, conn)

pp
# agrupando la frecuencia o veces que un jugador tiene máximas puntuaciones

# los mejores jugadores 

query = "SELECT overall_rating,count(overall_rating) as a FROM (SELECT distinct(player_api_id),overall_rating FROM Player_Attributes a  INNER JOIN (SELECT player_name, player_api_id AS p_id FROM Player) b ON a.player_api_id = b.p_id) where overall_rating > 88 group by overall_rating order by overall_rating desc;"



qq=pd.read_sql(query, conn)

qq
import numpy as np

import matplotlib.pyplot as plt



x = qq.overall_rating

y = qq.a

colors = qq.overall_rating*22

area = qq.a * qq.overall_rating  # np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii



plt.scatter(x, y, s=area, c=colors, alpha=0.5)

plt.show()
query = "SELECT * FROM Player;"

a = pd.read_sql(query, conn)

a.head()
query = "SELECT * FROM Player_Attributes;"

a = pd.read_sql(query, conn)

a.head()
query ="SELECT a.* FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_api_id=30657;"

b = pd.read_sql(query, conn)

b.head()
import matplotlib.pyplot as plt

plt.plot(b['date'].str[:4], b['agility'], marker='o', linestyle='-', color='r', label = "reflexes")

plt.plot(b['date'].str[:4], b['reactions'], marker='o', linestyle='-', color='b', label = "diving")

plt.plot(b['date'].str[:4], b['balance'], marker='o', linestyle='-', color='y',label = "kicking")

plt.plot(b['date'].str[:4], b['vision'], marker='o', linestyle='-', color='g',label = "positioning")

plt.plot(b['date'].str[:4], b['strength'], marker='o', linestyle='-', color='c',label = "handling")

plt.title("Evolución capacidades Messi") 

plt.xlabel("Años") 

plt.ylabel("Puntuación")

plt.legend()

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sqlite3

import seaborn as sns

from subprocess import check_output





query ="select * from Country;"

pd.read_sql(query, conn)





#otra visión



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sqlite3

import seaborn as sns

from subprocess import check_output



conn = sqlite3.connect('../input/database.sqlite')

cur = conn.cursor()

countries = cur.execute('select id,name from Country').fetchall()



plt.figure()

plt.hold(True)

county_names = []

Home_factor = np.zeros(len(countries))

home_dict = dict()

away_dict = dict()



for j,row in enumerate(countries):

    goals_home_list = []

    goals_away_list = []

    goals_home = cur.execute('select home_team_goal from Match where country_id =' + str(row[0])).fetchall()

    goals_away = cur.execute('select away_team_goal from Match where country_id =' + str(row[0])).fetchall()

    for i,game in enumerate(goals_home):

        goals_home_list.append(goals_home[:][i][0])

        goals_away_list.append(goals_away[:][i][0])

        

    Diff =  np.array(goals_home_list) - np.array(goals_away_list)

    Home_pct = np.true_divide(len(Diff[Diff>0]),len(Diff))

    Away_pct = np.true_divide(len(Diff[Diff<0]),len(Diff))

    Draw_pct = np.true_divide(len(Diff[Diff == 0]),len(Diff))

    

    away_expect = Away_pct*3 + Draw_pct

    home_expect = Home_pct*3 + Draw_pct

    

    home_dict[row[1]] = home_expect

    away_dict[row[1]] = away_expect



    if (row[1] == 'Spain') | (row[1] == 'Portugal') | (row[1] == 'France'):

       sns.distplot(Diff,hist = False,kde_kws={"shade": True})



    print(row[1], '   Home Win:', round(Home_pct,2), '   Draw:', round(Draw_pct,2),'   Away Win:', round(Away_pct,2), '   Average Difference:',round(np.mean(Diff),2))



plt.legend(['Portugal', 'España','Francia'], fontsize = 13)

plt.xlim([-10,10])

plt.title('Distribución de la diferencia de goles entre "en casa" versus "fuera"')

plt.show()
import pandas as pd

import sqlite3

conn = sqlite3.connect('../input/database.sqlite')



query = """SELECT * FROM Player_attributes a

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

players.head()
players = players.fillna(0)



cols = ['player_api_id','player_name','overall_rating','potential']

stats_cols = [col for col in players.columns if col not in (cols)]



from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

tmp = ss.fit_transform(players[stats_cols])

tmp
model = TSNE(n_components=2, random_state=0)

tsne_comp = model.fit_transform(tmp)
from bokeh.plotting import figure, ColumnDataSource, show

from bokeh.models import HoverTool



tmp = players[cols]

tmp['comp1'], tmp['comp2'] = tsne_comp[:,0], tsne_comp[:,1]

tmp = tmp[tmp.overall_rating >= 80]



_tools = 'box_zoom,pan,save,resize,reset,tap,wheel_zoom'

fig = figure(tools=_tools, title='t-SNE of Players (FIFA stats)', responsive=True,

             x_axis_label='Component 1', y_axis_label='Component 2')



source = ColumnDataSource(tmp)

hover = HoverTool()

hover.tooltips=[('Jogador','@player_name'),]

fig.scatter(tmp['comp1'], tmp['comp2'], source=source, size=8, alpha=0.6,

            line_color='red', fill_color='red')



fig.add_tools(hover)



show(fig)