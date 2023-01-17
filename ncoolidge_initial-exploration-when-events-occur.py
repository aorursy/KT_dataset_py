#Importing the libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # more plotting



#Importing the data

events = pd.read_csv('../input/events.csv')

ginf = pd.read_csv('../input/ginf.csv')
events.info() # First, let's take a look at what kind of info we have.
# I manually converted the ../input/dictionary.txt to python dicts

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
# Notice that a lot of the objects are in fact categoricals, which are much easier to work with

# We can fix this with Pandas' astype function

cats = ['id_odsp', 'event_type', 'player', 'player2', 'event_team', 'opponent', 'shot_place', 'shot_outcome', 'location', 'bodypart', 'assist_method', 'situation']

d = dict.fromkeys(cats,'category')

events = events.astype(d)

events['is_goal'] = events['is_goal'].astype('bool') # this is a bool, we can fix that too while we're at it

events.info() # much better
goals = events[events['is_goal'] == True] # events where a goal resulted
# When do the goals occur?

plt.hist(goals.time, 100)

plt.xlabel("Time")

plt.ylabel("Number of Goals at Time")

plt.title("When Goals Occur")

plt.show()
ninetiethMin = goals[goals['time'] == 90]

eightyNinthMin =  goals[goals['time'] == 89]

ninetiethMin['dupes'] = ninetiethMin['id_odsp'].duplicated(keep=False)

eightyNinthMin['dupes'] = eightyNinthMin['id_odsp'].duplicated(keep=False)

multiNinetiethGoals = ninetiethMin[ninetiethMin['dupes'] == True]

multiEightyNinthGoals = eightyNinthMin[eightyNinthMin['dupes'] == True]

# I'm ignoring this warning for now, because it doesn't really matter for our purposes.
print("There were " + str(len(ninetiethMin)) + " goals in the 90th minute vs. " + str(len(eightyNinthMin)) + " in the 89th.\n" + str(len(multiNinetiethGoals)) + " times, there were multiple goals in the 90th minute, vs. " + str(len(multiEightyNinthGoals)) + " multiples in the 89th.")
yellowCards = events[events['event_type'] == ('Yellow card' or 'Second yellow card')] # selects yellow cards

redCards = events[events['event_type'] == 'Red card'] # selects red cards

substitutions = events[events['event_type'] == 'Substitution'] # selects substitutions
# When do the substitutions occur?

plt.hist(substitutions.time, 100)

plt.xlabel("Time")

plt.ylabel("Substitutions")

plt.title("When Substitutions Occur")

plt.show()
# When do the red cards occur?

reds = plt.hist(redCards.time, 100, color="red")

plt.xlabel("Time")

plt.ylabel("Red Cards")

plt.title("When Red Cards Occur")

plt.show()
# Or, breaking it into slightly larger chunks, which seems to paint a different picture

reds = plt.hist(redCards.time, 10, color="red")

plt.xlabel("Time")

plt.ylabel("Red Cards")

plt.title("When Red Cards Occur")

plt.show()
# When do the yellow cards occur?

yellows = plt.hist(yellowCards.time, 100, color="yellow")

plt.xlabel("Time")

plt.ylabel("Yellow Cards")

plt.title("When Yellow Cards Occur")

plt.show()
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