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
import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from matplotlib.pyplot import figure

import statistics
injuryrec = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

injuryrec.head()
plt.figure(figsize=(8,8))

injuries = sns.countplot(x = 'BodyPart', data = injuryrec)

plt.title('Injured Body Parts')

plt.show()
# plot of FieldTyp per Indoor & Outdoor Game 

plt.figure(figsize=(15,5))

sns.countplot(x = 'BodyPart', data = injuryrec, hue = 'Surface')

plt.title('Injured Body Parts by Surfaces')

plt.xlabel('Injured Body Parts')

plt.ylabel('Number of Injuries')

plt.show()
l = [injuryrec['DM_M1'].count()]  # list of the player unjured per time point

l.append(injuryrec['DM_M7'].value_counts()[1])

l.append(injuryrec['DM_M28'].value_counts()[1])

l.append(injuryrec['DM_M42'].value_counts()[1])





l2 = []

for i in range(0, len(l)-1):

        l2.append(l[i] - l[i+1])

l2.append(injuryrec['DM_M42'].value_counts()[1])

#-----------------------------------------------------for the graphic -----------------------------------------------------

l3 = ['1 day', '7 days', '28 days', '42 days'] # label

plt.figure(figsize=(10,5))

plt.title('How many players are unable to play for how long?') #title

plt.ylabel('Numbers of players')

plt.bar(l3, l2)

plt.show()
playlist = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

playlist.head()
plt.figure(figsize=(18,5))

sns.countplot(x = 'Temperature', data = playlist)

plt.xlabel('Temperature in °F')

plt.ylabel('Number of Games')

plt.show()
#stadium = playlist.groupby('StadiumType').count()

plt.figure(figsize=(10,10))

sns.countplot(y = 'StadiumType', data = playlist) # the plot is better visible when turned! 

plt.xlabel('Number of Games')

plt.ylabel('Types of Stadium')

plt.show()
# cleaning the StadiumType column:

d = {'Oudoor': 'Outdoor', 'Outdoors': 'Outdoor', 'Ourdoor': 'Outdoor', 'Outdor': 'Outdoor', 'Outside': 'Outdoor', 'Indoors' : 'Indoor', 'Outddors' : 'Outdoor', \

     'Domed, closed': 'Closed Dome', 'Domed': 'Dome', 'Domed, Open':'Open Dome', 'Dome, closed' : 'Closed Dome', 'Retr. Roof-Closed' : 'Retr. Roof Closed',\

     'Retr. Roof - Closed': 'Retr. Roof Closed', 'Outdoor Retr Roof-Open': 'Retr. Roof-Open', 'Indoor, Open Roof' : 'Retr. Roof-Open', 'Indoor, Roof Closed': 'Retr. Roof Closed',\

     'Retr. Roof - Open': 'Retr. Roof-Open','Heinz Field' : 'Outdoor'}

playlist_mod = playlist.replace(d)



playlist_mod['StadiumType'].unique() #checking for unique categories
plt.figure(figsize=(10,10))

sns.countplot(y = 'StadiumType', data = playlist_mod) # the plot is better visible when turned! 

plt.xlabel('Number of Games')

plt.ylabel('Types of Stadium')

plt.show()
d2 = {'Open' : 'Outdoor', 'Closed Dome' : 'Indoor', 'Dome' : 'Indoor', 'Retr. Roof Closed' : 'Indoor', 'Retr. Roof-Open' : 'Outdoor', 'Retractable Roof' : 'Indoor', 'Open Dome' : 'Outdoor', 'Domed, open' : 'Outdoor' }

playlist_mod2 = playlist_mod.replace(d2)
# drop lines with a certain content from a column

indexNames = playlist_mod2[playlist_mod2['StadiumType'] == 'Bowl'].index

playlist_mod2.drop(indexNames, inplace = True)

indexNames = playlist_mod2[playlist_mod2['StadiumType'] == 'Cloudy'].index

playlist_mod2.drop(indexNames, inplace = True)

playlist_mod2['StadiumType'].unique()
plt.figure(figsize=(8,8))

sns.countplot(x = 'StadiumType', data = playlist_mod2) # the plot is better visible when turned! 

plt.ylabel('Number of Games')

plt.xlabel('Types of Stadium')

plt.show()
# plot of FieldTyp per Indoor & Outdoor Game 

plt.figure(figsize=(20,8))

sns.countplot(x = 'StadiumType', data = playlist_mod2, hue = 'FieldType')

plt.title('Distribution of Synthetic and Natural turf in games played indoor or outdoor')

plt.xlabel('Field types')

plt.ylabel('Number of Games')

plt.show()
plt.figure(figsize=(15,15))

sns.countplot(y = 'Weather', data = playlist) # the plot is better visible when turned! 

plt.title('Recorded weather categories ')

plt.xlabel('Number of Games')

plt.ylabel('Weather conditions')

plt.show()
plt.figure(figsize=(20,5))

sns.countplot(x = 'RosterPosition', data = playlist) # the plot is better visible when turned! 

plt.title('Player Positions')

plt.xlabel('Positions')

plt.ylabel('Number of records')

plt.show()
injuryrec.head()
playlist_mod.head()
inplay = pd.merge(injuryrec, playlist_mod, on = 'PlayKey')

inplay.info()
inplay.tail()
plt.figure(figsize=(15,5))

sns.countplot(x = 'PlayType', data = inplay, hue = 'BodyPart') 

plt.title('Injuries per Play Type')

plt.xlabel('Play Type')

plt.ylabel('Number of injuries')

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(x = 'PositionGroup', data = inplay, hue = 'BodyPart') 

plt.title('Injuries per Position Group')

plt.xlabel('Position')

plt.ylabel('Number of injuries')

plt.show()
plt.figure(figsize=(15,8))

sns.countplot(x = 'Position', data = inplay, hue = 'BodyPart') 

plt.title('Injuries per Position Group')

plt.xlabel('Position')

plt.ylabel('Number of injuries')

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(x = 'PositionGroup', data = inplay, hue = 'Surface') 

plt.title('Injuries per Position Group')

plt.xlabel('Position Group')

plt.ylabel('Number of injuries')

plt.figure(figsize=(15,5))

sns.countplot(x = 'Position', data = inplay, hue = 'Surface') 

plt.title('Injuries per Position')

plt.xlabel('Position')

plt.ylabel('Number of injuries')

plt.show()
plt.figure(figsize=(15,6))

sns.countplot(x = 'BodyPart', data = inplay, hue = 'Weather') 

plt.title('Injuried Body Part')

plt.xlabel('Body Part')

plt.ylabel('Number of injuries')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # putting the legend outside of the figure

plt.figure(figsize=(15,10))

sns.countplot(x = 'BodyPart', data = inplay, hue = 'Temperature') 

plt.title('Injuried Body Part by Temperature')

plt.xlabel('Body Part')

plt.ylabel('Number of injuries')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # putting the legend outside of the figure

plt.figure(figsize=(15,10))

sns.countplot(x = 'Surface', data = inplay, hue = 'Temperature') 

plt.title('Surface by Temperature')

plt.xlabel('Surface')

plt.ylabel('Number of injuries')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # putting the legend outside of the figure

plt.show()
player = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')

player.head()
total = pd.merge(inplay, player, on = 'PlayKey')

total.head()
fig = px.scatter_matrix(total,

    dimensions=['Surface', 's'],

    color="BodyPart")

fig.show()