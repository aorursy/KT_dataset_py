import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

sns.set_style('darkgrid')

mpl.rcParams['figure.figsize'] = [15,10]
match = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
match.head()
home_players_id = list(match['DisplayName'].to_list())[:11]

away_players_id = list(match['DisplayName'].to_list())[11:22]

home_players_x = list(match['X'].to_list())[:11]

away_players_x = list(match['X'].to_list())[11:22]

home_players_y = list(match['Y'].to_list())[:11]

away_players_y = list(match['Y'].to_list())[11:22]







print('Example, home players id: ')

print(home_players_id)
import matplotlib.pyplot as plt



# Home team (in blue)

plt.subplot(2, 1, 1)

plt.rc('grid', linestyle="-", color='black')

plt.rc('figure', figsize=(12,20))

plt.gca().invert_yaxis() # Invert y axis to start with the goalkeeper at the top

for label, x, y in zip(home_players_id, home_players_x, home_players_y):

    plt.annotate(

        label, 

        xy = (x, y), xytext = (-20, 20),

        textcoords = 'offset points', va = 'bottom')

plt.scatter(home_players_x, home_players_y,s=280,c='blue')

plt.grid(True)



# Away team (in red)

plt.subplot(2, 1, 2)

plt.rc('grid', linestyle="-", color='black')

plt.rc('figure', figsize=(12,20))

plt.gca().invert_xaxis() # Invert x axis to have right wingers on the right

for label, x, y in zip(away_players_id, away_players_x, away_players_y):

    plt.annotate(

        label, 

        xy = (x, y), xytext = (-20, 20),

        textcoords = 'offset points', va = 'bottom')

plt.scatter(away_players_x, away_players_y,s=280,c='red')

plt.grid(True)





ax = [plt.subplot(2,2,i+1) for i in range(0)]

for a in ax:

    a.set_xticklabels([])

    a.set_yticklabels([])

#plt.subplots_adjust(wspace=0, hspace=0)





plt.show()
from collections import Counter



players_y = [home_players_y,away_players_y]

formations = [None] * 2

for i in range(2):

    formation_dict=Counter(players_y[i]);

    sorted_keys = sorted(formation_dict)

    formation = ''

    for key in sorted_keys[1:-1]:

        y = formation_dict[key]

        formation += '%d-' % y

    formation += '%d' % formation_dict[sorted_keys[-1]] 

    formations[i] = formation





print('Home team formation: ' + formations[0])

print('Away team formation: ' + formations[1])