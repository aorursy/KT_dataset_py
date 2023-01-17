# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
teams=pd.read_csv('../input/ipl-data-set/teams.csv')
teams
team_h_a=pd.read_csv('../input/ipl-data-set/teamwise_home_and_away.csv')
team_h_a
matches=pd.read_csv('../input/ipl-data-set/matches.csv')
matches.head()
matches.shape
matches[:][matches['Season']=='IPL-2019']
matches_in_2019=matches[:][matches['Season']=='IPL-2019'].shape[0]
## Team winning most matches in the ipl 2019

matches['winner']
import statistics
statistics.mode(matches['winner'])
statistics.mode(matches['toss_winner'])
won_who_won_the_toss=matches[:][matches['Season']=='IPL-2019'][matches['toss_winner']==matches['winner']].shape[0]
team_who_won_the_toss_winning=(won_who_won_the_toss/matches_in_2019)*100
team_who_won_the_toss_winning
matches[:][matches['Season']=='IPL-2019'][matches['toss_winner']==matches['winner']][matches['toss_decision']=='bat']
matches[:][matches['Season']=='IPL-2019'][matches['toss_winner']==matches['winner']][matches['toss_decision']=='bat'].shape[0]
((matches[:][matches['Season']=='IPL-2019'][matches['toss_winner']==matches['winner']][matches['toss_decision']=='bat'].shape[0])/(matches_in_2019))*100
ipl_2019=matches[:][matches['Season']=='IPL-2019']
ipl_2019
statistics.mode(ipl_2019['player_of_match'])
from scipy import stats as s

print((s.mode(ipl_2019['venue'])[0][0]))
print((s.mode(ipl_2019['city'])[0][0]))
teams=[x for x in (set(ipl_2019['team1']))]
teams
ipl_2019['winner'].values
c=0

win=[]

for i in teams:

    for j in ipl_2019['winner'].values:

        if i==j:

            c=c+1

    win.append(c)

    c=0
win
team_score=pd.DataFrame({'Teams':teams,'win':win})
team_score
c=0

win=[]

for i in teams:

    for j in matches['winner'].values:

        if i==j:

            c=c+1

    win.append(c)

    c=0
team_score=pd.DataFrame({'Teams':teams,'win':win})
team_score
c=0

win=[]

for i in teams:

    for j in matches['toss_winner'].values:

        if i==j:

            c=c+1

    win.append(c)

    c=0
team_score=pd.DataFrame({'Teams':teams,'Toss_win':win})
team_score
players=[x for x in (set(matches['player_of_match']))]
c=0

win=[]

for i in players:

    for j in matches['player_of_match'].values:

        if i==j:

            c=c+1

    win.append(c)

    c=0
team_score=pd.DataFrame({'Player':players,'Player_of_match':win})
team_score
team_score[:][team_score['Player_of_match']>10]