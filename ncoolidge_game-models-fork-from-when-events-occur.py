#Importing the libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # cool plotting

from scipy import stats

#Importing the data

events = pd.read_csv('../input/events.csv')

ginf = pd.read_csv('../input/ginf.csv')
# I manually converted the ../input/dictionary.txt to Python dicts. I'm sure that there's a better way to do that, but this works.

event_types = {1:'Attempt', 2:'Corner', 3:'Foul', 4:'Yellow card', 5:'Second yellow card', 6:'Red card', 7:'Substitution', 8:'Free kick won', 9:'Offside', 10:'Hand ball', 11:'Penalty conceded'}

event_types2 = {12:'Key Pass', 13:'Failed through ball', 14:'Sending off', 15:'Own goal'}

sides = {1:'Home', 2:'Away'}

shot_places = {1:'Bit too high', 2:'Blocked', 3:'Bottom left corner', 4:'Bottom right corner', 5:'Centre of the goal', 6:'High and wide', 7:'Hits the bar', 8:'Misses to the left', 9:'Misses to the right', 10:'Too high', 11:'Top centre of the goal', 12:'Top left corner', 13:'Top right corner'}

shot_outcomes = {1:'On target', 2:'Off target', 3:'Blocked', 4:'Hit the bar'}

locations = {1:'Attacking half', 2:'Defensive half', 3:'Centre of the box', 4:'Left wing', 5:'Right wing', 6:'Difficult angle and long range', 7:'Difficult angle on the left', 8:'Difficult angle on the right', 9:'Left side of the box', 10:'Left side of the six yard box', 11:'Right side of the box', 12:'Right side of the six yard box', 13:'Very close range', 14:'Penalty spot', 15:'Outside the box', 16:'Long range', 17:'More than 35 yards', 18:'More than 40 yards', 19:'Not recorded'}

bodyparts = {1:'right foot', 2:'left foot', 3:'head'}

assist_methods = {0:np.nan, 1:'Pass', 2:'Cross', 3:'Headed pass', 4:'Through ball'}

situations = {1:'Open play', 2:'Set piece', 3:'Corner', 4:'Free kick'}
# Mapping the dicts onto the events dataframe

events['event_type'] = events['event_type'].map(event_types)

events['event_type2'] = events['event_type2'].map(event_types2)

events['side'] = events['side'].map(sides)

events['shot_place'] = events['shot_place'].map(shot_places)

events['shot_outcome'] = events['shot_outcome'].map(shot_outcomes)

events['location'] = events['location'].map(locations)

events['bodypart'] = events['bodypart'].map(bodyparts)

events['assist_method'] = events['assist_method'].map(assist_methods)

events['situation'] = events['situation'].map(situations)
# A few convertions to bool and datetime, respectively

events = events.astype({'is_goal':'bool', 'fast_break':'bool'})

ginf['date'] = pd.to_datetime(ginf['date'])
# It will be helpful to know if the team who caused that event won the game. Takes a while because it goes through events row-wise.

def defineWinner(row):

    if row['fthg'] > row['ftag']:

        row['result'] = 'Home win'

    elif row['ftag'] > row['fthg']:

        row['result'] = 'Away win'

    elif row['fthg'] == row['ftag']:

        row['result'] = 'Draw'

    else: # For when scores are missing, etc (should be none)

        row['result'] = None

    return row

ginf = ginf.apply(defineWinner, axis=1)
# Merging the two dataframes, leaving only games where we have event information.

gameEvents = pd.merge(events, ginf, on='id_odsp', how='left') 
# a few more conversions - preferable to use 'category' rather than 'object' when dealing with multiples of the same object

cats = ['event_type', 'player', 'player2', 'event_team', 'opponent', 'shot_place', 'shot_outcome', 'location', 'bodypart', 'assist_method', 'situation', 'league', 'season', 'country', 'ht', 'at', 'result']

d = dict.fromkeys(cats,'category')

gameEvents = gameEvents.astype(d)
home_count = gameEvents['fthg'].value_counts(sort = False, bins = 11)

away_count = gameEvents['ftag'].value_counts(sort = False, bins = 11)

fig, ax = plt.subplots()

rects1 = ax.bar(np.arange(11), home_count, 0.35, color='b')

rects2 = ax.bar(np.arange(11) + 0.35, away_count, 0.35, color='r')



# add some text for labels, title and axes ticks

ax.set_ylabel('Goals')

ax.set_title('Home Goals vs Away Goals')

ax.set_xticks(np.arange(11) + 0.35 / 2)

ax.set_xticklabels(np.arange(11))



ax.legend((rects1[0], rects2[0]), ('Home goals', 'Away goals'))



plt.show()