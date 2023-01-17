# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import sqlite3



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
con = sqlite3.connect('../input/database.sqlite')
country = pd.read_sql_query("SELECT * FROM Country", con)

league = pd.read_sql_query("SELECT * FROM League", con)

match = pd.read_sql_query("SELECT * FROM Match", con)



league.head()
leagueMatch = pd.merge(league, match, left_on='id', right_on='league_id',how='left')
leagueMatch.info()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(leagueMatch.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
leagueMatch.head()
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

leagueMatch.home_team_goal.plot(kind = 'line', color = 'g',label = 'Home goal',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

leagueMatch.away_team_goal.plot(color = 'r',label = 'Away goal',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

leagueMatch.plot(kind='scatter', x='home_team_goal', y='away_team_goal',alpha = 0.5,color = 'red')

plt.xlabel('Home goal') # label = name of label

plt.ylabel('Away goal')

plt.title('Difference berween home and away goals')

plt.show()
# Histogram

# bins = number of bar in figure

leagueMatch.home_team_goal.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.title('Home goals')

plt.show()



leagueMatch.away_team_goal.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.title('Away goals')

plt.show()
leagues = leagueMatch['name']

print(type(leagues))

leagues.head()
f_leagues = leagueMatch[['name']]

print(type(f_leagues))

f_leagues.head()
homeGoals = leagueMatch['home_team_goal'] > 7

leagueMatch[homeGoals]
leagueMatch[(leagueMatch['home_team_goal']>7) & (leagueMatch['away_team_goal']>1)]
for index,value in leagueMatch[['home_team_goal']][0:4].iterrows():

    print(index," : ",value)