# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv("../input/fifa19/data.csv")

equipos= ('Stevenage', 'Coventry City', 'Morecambe', 'Wycombe Wanderers', 'Cambridge United',

             'Exeter City', 'Luton Town', 'Notts County', 'Macclesfield Town')

clubes = data.loc[data['Club'].isin(equipos) & data['Overall']]

puntajes_league_2=sns.boxplot(x =clubes ['Club'], y =clubes ['Overall'], palette = 'Paired')

puntajes_league_2.set_xlabel(xlabel = 'Equipos de la Football League', fontsize = 12)

puntajes_league_2.set_ylabel(ylabel = 'Puntuación de los jugadores', fontsize = 12)

puntajes_league_2.set_title(label = 'Distribución de los puntajes de equipos de la Football League (League 2)', fontsize = 20)



plt.xticks(rotation = 90)

plt.show()



equipos= ('Stevenage', 'Coventry City', 'Morecambe', 'Wycombe Wanderers', 'Cambridge United',

             'Exeter City', 'Luton Town', 'Notts County', 'Macclesfield Town')

clubes = data.loc[data['Club'].isin(equipos) & data['Overall']]

puntajes_league_2=sns.barplot(x =clubes ['Club'], y =clubes ['Overall'], palette = 'Paired')

puntajes_league_2.set_xlabel(xlabel = 'Equipos de la Football League', fontsize = 12)

puntajes_league_2.set_ylabel(ylabel = 'Puntuación de los jugadores', fontsize = 12)

puntajes_league_2.set_title(label = 'Puntajes de equipos de la Football League (League 2)', fontsize = 20)



plt.xticks(rotation = 90)

plt.show()


jugadores = ('E. Hazard', 'Sergio Ramos', 'Miranda', 'Maicon', 'Allan',

             'Parejo','Liu Yue')

clubes = data.loc[data['Name'].isin(jugadores) & data['Overall']]

puntajes_jugadores=sns.barplot(x =clubes ['Name'], y =clubes ['Overall'], palette = 'inferno')

puntajes_jugadores.set_xlabel(xlabel = 'Jugadores del FIFA 19', fontsize = 12)

puntajes_jugadores.set_ylabel(ylabel = 'Puntuación de los jugadores', fontsize = 12)

puntajes_jugadores.set_title(label = 'Puntuaciones de jugadores en el FIFA 19', fontsize = 20)

plt.xticks(rotation = 90)



plt.show()


paises = ( 'Germany', 'Wales', 'Hungary', 'Serbia',

             'England','Finland')

pais_datos = data.loc[data['Nationality'].isin(paises) & data['Age']]



edad_paises=sns.boxenplot(x =pais_datos ['Nationality'], y =pais_datos ['Age'], palette = 'Set2')

edad_paises.set_xlabel(xlabel = 'Edad de jugadores', fontsize = 12)

edad_paises.set_ylabel(ylabel = 'Países', fontsize = 12)

edad_paises.set_title(label = 'Edad de jugadores en el FIFA 19 por países', fontsize = 20)



plt.xticks(rotation = 90)

plt.show()


eq = ('Chelsea', 'Arsenal', 'AC Milan',

             'Juventus','Wycombe Wanderers')

equipos3_datos = data.loc[data['Club'].isin(eq) & data['International Reputation']]



rep_equipos=sns.violinplot(x =equipos3_datos ['Club'], y = equipos3_datos ['International Reputation'], palette = 'Paired')

rep_equipos.set_xlabel(xlabel = 'Clubes', fontsize = 12)

rep_equipos.set_ylabel(ylabel = 'Reputación Internacional', fontsize = 12)

rep_equipos.set_title(label = 'Reputación jugadores del FIFA 19', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
