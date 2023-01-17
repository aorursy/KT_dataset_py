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
os.getcwd()

df = pd.read_csv('/kaggle/input/international-football-results-from-1872-to-2017/results.csv')

df['result'] = ['Draw' if i == 0  else 'Home_Won' if i > 0  else 'Away_Won' for i in df['home_score'] - df['away_score']]
df.head(10)


team_list = list(df['home_team'].unique())

team_list.extend(list(df['away_team'].unique()))

team_list = list(set(team_list))

team_goals_dict = {}

total_goals = []

total_games = []

total_conceded = []

total_win = []

total_lost = []



for team in team_list:

    total_goals.append(sum(df[df['home_team'] == team]['home_score']) + sum(df[df['away_team'] == team]['away_score']))

    total_games.append(len(df[df['home_team'] == team]) + len(df[df['away_team'] == team]))

    total_conceded.append(sum(df[df['home_team'] == team]['away_score']) + sum(df[df['away_team'] == team]['home_score']))

    total_win.append(len(df[(df['home_team'] == team) & (df['result'] == 'Home_Won')]) + len(df[(df['away_team'] == team) & (df['result'] == 'Away_Won')]))

    total_lost.append(len(df[(df['home_team'] == team) & (df['result'] == 'Away_Won')]) + len(df[(df['away_team'] == team) & (df['result'] == 'Home_Won')]))



team_goals_dict['teams'] = team_list

team_goals_dict['total_goals'] = total_goals

team_goals_dict['total_games'] = total_games

team_goals_dict['total_conceded'] = total_conceded

team_goals_dict['total_win'] = total_win

team_goals_dict['total_lost'] = total_lost
#print(team_goals_dict)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



df_tally = pd.DataFrame(team_goals_dict)

df_tally['total_draw'] = df_tally['total_games'] - df_tally['total_win'] - df_tally['total_lost'] 

df_tally['goals_per_game'] = df_tally['total_goals'] / df_tally['total_games']

df_tally['goals_conceded_per_game'] = df_tally['total_conceded'] / df_tally['total_games']

df_goal_top = df_tally.sort_values(ascending=False,by=['total_goals']).reset_index(drop=True).head(10)

df_goal_top.head(10)
df_goal_per_game_top = df_tally[df_tally['total_games'] > 500].sort_values(ascending=False,by=['goals_per_game']).reset_index(drop=True).head(10)

df_goal_per_game_top 
sns.set(rc={'figure.figsize':(15,12)})

sns.barplot(x='teams',y='total_goals',data =df_goal_top)
import plotly.express as plty

fig = plty.bar(df_goal_per_game_top,x='teams',y='goals_per_game',color = 'total_games',color_continuous_scale='Viridis')

fig.show()
df_worst_defending = df_tally.sort_values(ascending=False,by=['total_conceded']).reset_index(drop=True).head(10)

df_worst_defending
sns.set(rc={'figure.figsize':(15.7,12.27)})

sns.barplot(x='teams',y='total_conceded',data =df_worst_defending)
df_goal_conceded_per_game_top = df_tally[df_tally['total_games'] > 500].sort_values(ascending=False,by=['goals_conceded_per_game']).reset_index(drop=True).head(10)

df_goal_conceded_per_game_top 
fig = plty.bar(df_goal_conceded_per_game_top,x='teams',y='goals_conceded_per_game',color = 'total_games',color_continuous_scale='Viridis')

fig.show()