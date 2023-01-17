import sqlite3

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



database = '../input/database.sqlite'

conn = sqlite3.connect(database)



query = "SELECT name as Tablas FROM sqlite_master WHERE type='table';"

pd.read_sql(query, conn)
jugador = pd.read_sql("SELECT * FROM Player", conn)

atributos_jugador = pd.read_sql("SELECT * FROM Player_attributes", conn)
jugador.head()
atributos_jugador.head()
atributos_jugador.describe()
query = "SELECT DISTINCT p.player_name, a.overall_rating FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id ORDER BY a.overall_rating DESC;"

b = pd.read_sql(query, conn)

b.head(10)
query = "SELECT a.date, a.overall_rating, p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_name='Lionel Messi';"

lio = pd.read_sql(query, conn)

lio.head()
query = "SELECT a.date, a.overall_rating, p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_name='Cristiano Ronaldo';"

cr7 = pd.read_sql(query, conn)

cr7.head()
query = "SELECT a.date, a.overall_rating, p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_name='Gianluigi Buffon';"

gb = pd.read_sql(query, conn)

gb.head()
query = "SELECT a.date, a.overall_rating, p.player_name FROM Player_Attributes a, player p where p.player_api_id=a.player_api_id and p.player_name='Wayne Rooney';"

wr = pd.read_sql(query, conn)

wr.head()
import matplotlib.pyplot as plt

plt.plot(lio['date'].str[:4], lio['overall_rating'], marker='o', linestyle='-', color='r')

plt.plot(cr7['date'].str[:4], cr7['overall_rating'], marker='x', linestyle='-', color='b')

plt.plot(gb['date'].str[:4], gb['overall_rating'], marker='v', linestyle='-', color='g')

plt.plot(wr['date'].str[:4], wr['overall_rating'], marker='v', linestyle='-', color='orange')

plt.show()
query = "SELECT p.player_name, a.date=max(a.date), a.* FROM Player_Attributes a, player p WHERE p.player_api_id=a.player_api_id and p.player_name='Lionel Messi';"

lio_att = pd.read_sql(query, conn)

lio_att
query = "SELECT p.player_name, a.date=max(a.date), a.* FROM Player_Attributes a, player p WHERE p.player_api_id=a.player_api_id and p.player_name='Cristiano Ronaldo';"

cr7_att = pd.read_sql(query, conn)

cr7_att
cols=['crossing', 'finishing', 'heading_accuracy','short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'positioning', 'vision', 'penalties']

lio_att=lio_att.loc[:,cols]

lio_att.head()
cr7_att=cr7_att.loc[:,cols]

cr7_att.head()
lio=tuple(lio_att.loc[0,:])

cr7=tuple(cr7_att.loc[0,:])



fig, ax = plt.subplots()

index = np.arange(23)

bar_width = 0.35

opacity = 0.8

rects1 = plt.bar(index, lio, bar_width, alpha=opacity, color='r', label='L.Messi')

rects2 = plt.bar(index+bar_width, cr7, bar_width, alpha=opacity, color='b', label='C.Ronaldo')

plt.xlabel('Attributes', fontsize=14)

plt.ylabel('Scores', fontsize=14)

plt.title('Scores by player', fontsize=18)

plt.xticks(index + bar_width, ('crossing', 'finishing', 'heading_accuracy','short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'positioning', 'vision', 'penalties'), fontsize=14, rotation=90)

plt.legend()

plt.tight_layout()

fig.set_size_inches(18.5, 16.5)

plt.show()
LMvsCR=[x-y for x,y in zip(lio, cr7)]

j=0

for i in LMvsCR:

    if i>0:

        LMvsCR[j]="Messi"

    elif i<0:

        LMvsCR[j]="Ronaldo"

    else:

        LMvsCR[j]="Empate"

    j=j+1

    

labels = ['Messi', 'Ronaldo', 'Empate']

porcentajes=[LMvsCR.count(i)/len(LMvsCR) for i in labels]

colors = ['r', 'b', 'grey']

plt.pie(porcentajes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True) 

plt.title('Messi vs Ronaldo: % de atributos que lidera cada uno')

plt.axis('equal')

plt.show()