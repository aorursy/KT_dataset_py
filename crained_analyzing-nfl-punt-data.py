import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

import os

import seaborn as sns

sns.set_style('whitegrid')



# For interactive plots

from plotly import offline

import plotly.graph_objs as go





pd.set_option('max.columns', None)

offline.init_notebook_mode()

config = dict(showLink=False)
# Read the input data

ppd = pd.read_csv('../input/player_punt_data.csv')

gd = pd.read_csv('../input/game_data.csv')

pprd = pd.read_csv('../input/play_player_role_data.csv')

vr = pd.read_csv('../input/video_review.csv')

vfi = pd.read_csv('../input/video_footage-injury.csv')

pi = pd.read_csv('../input/play_information.csv')
# I want to create another variable that is concussive_games so that I can leave vr for later

# I'm going to build on this variable

concussive_games = vr

concussive_games.head()
# Here I'm going to merge games where someone got a concussion with the game data

# game data has a lot more information: 

# Game level data that specifies the type of season (pre, reg, post), week, and hosting city and team. 

# Each game is uniquely identified across all seasons using GameKey.

conGD = pd.merge(concussive_games, gd)

conGD.head()
# lets verify we still have only 37 games to deal with

conGD.describe()
#lets look at concussions by season year

sns.factorplot('Season_Year',data=conGD,kind='count')
# lets look at the season year but by season type: preseason, regular season and post season

sns.factorplot('Season_Year',data=conGD,hue='Season_Type', kind='count')
# let's look play information

# this gives specific detail to the play

pi.tail()
pi.describe()
# lets look at the punts by season year and season type: preseason, regular season and post season

sns.factorplot('Season_Year',data=pi,hue='Season_Type', kind='count')
# after testing different combinations I found that there is duplicate GSISID data

# however the player number was different so I'm looking at it here. 

ppd.head(20).sort_values('GSISID')
# here I want see what is duplicated

ppd[ppd.duplicated(['GSISID']) == True].sort_values('GSISID')
# FIND 33941 to see why it is duplicating when brought into concussive plays

ppd.loc[ppd['GSISID'] == 33941]
#remove the letters from the Numbers column

import re

for Number in ppd:

    ppd['Number'] = [''.join(re.findall("\d*\.?\d+", item)) for item in ppd['Number']]
# Lets see if we removed the characters from the Numbers column

ppd.loc[ppd['GSISID'] == 33941]
# here I want see what is duplicated after cleanup

ppd[ppd.duplicated(['GSISID']) == True]
#There are still issues. Such as a 0 before 3, we need to remove that. 

ppd['Number'] = ppd['Number'].str.replace('0','')
# Lets see if we removed the characters from the Numbers column

ppd.loc[ppd['GSISID'] == 33941]
# After all that cleanup it is clear that numbrer is meaningless so lets remove the jersey number

ppdDrop = ppd.drop('Number', 1)
ppdDrop.head()
# merge games where someone got a concussion with the game data

# and merged the player data in hopes that we still get 38

conPlayer = pd.merge(conGD, ppdDrop)

conPlayer.describe()
conPlayer[conPlayer.duplicated(['PlayID','GSISID']) == True]
conPlayer = conPlayer.dropna(subset=['PlayID']).drop_duplicates(['PlayID','GSISID'])
conPlayer.describe()
sns.factorplot('Position',data=conPlayer,kind='count')
sns.catplot(data=conPlayer, x="Position",col="Player_Activity_Derived", kind="count")
sns.catplot('Primary_Partner_Activity_Derived',data=conPlayer,kind='count')
sns.catplot(data=conPlayer, x="Position",col="Primary_Partner_Activity_Derived", kind="count")
#conPlayer has primary partner GSISID

# ppdDrop has the player positions

conPlayer = conPlayer.rename(columns={'Position': 'Concussed_Position'})

conPlayer.head()
#NAme Position - Concussed Position

#Name Position - Enemy Position

conPlayer1 = conPlayer.rename(columns={'GSISID': 'Concussed_GSISID'})

conPlayer1.head()
conPlayer2 = conPlayer1.rename(columns={'Primary_Partner_GSISID': 'GSISID'})

conPlayer2.head()
# dropping NAN from GSISID because some or NAN becuase the ground can cause a concussion

conPlayer2['GSISID'] = pd.to_numeric(conPlayer2['GSISID'], errors='coerce')

conPlayer2 = conPlayer2.dropna(subset=['GSISID'])

conPlayer2['GSISID']=conPlayer2['GSISID'].apply(int)

conPlayer2.head()
conPlayer2.describe()
# merged enemy player with their ID and position. They caused the concussion

conPlayer3 = pd.merge(conPlayer2, ppdDrop)

conPlayer3.head(20)
conPlayer3.describe()
conPlayer4 = conPlayer3.dropna(subset=['PlayID']).drop_duplicates(['PlayID','Concussed_GSISID','GSISID'])
conPlayer4.describe()
# convert GSISID to Enemy_GSISID

conPlayer5 = conPlayer4.rename(columns={'GSISID': 'Enemy_GSISID'})

conPlayer5.head()
# convert Position to Enemy_Position and drop number

conPlayer5 = conPlayer4.rename(columns={'Position': 'Enemy_Position'})

#conPlayer5 = conPlayer4.drop('Number', 1)

conPlayer5.head()
conPlayer5.describe()
sns.catplot(data=conPlayer5, x="Concussed_Position",col="Enemy_Position", kind="count", col_wrap=4 )
sns.catplot(data=conPlayer5, x="Primary_Partner_Activity_Derived",col="Enemy_Position", kind="count", col_wrap=4)
sns.catplot("Primary_Impact_Type", data=conPlayer5,kind="count")

#sns.catplot('Primary_Partner_Activity_Derived',data=conPlayer,kind='count')
conPlayer5.head()