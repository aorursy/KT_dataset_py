# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline



import matplotlib

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
wnba_players = pd.read_csv('../input/wnba-player-stats-2017/WNBA Stats.csv')

wnba_players.fillna(0,inplace=True)

sns.boxplot(x='Pos', y='Height', data=wnba_players)

sns.swarmplot(x='Pos', y='Height', data=wnba_players, color='0')

plt.title('Height of Players per position (WNBA)')

plt.xlabel('Position')

plt.ylabel('Height (cm)')
sns.distplot(wnba_players['Height'])
sns.distplot(wnba_players[wnba_players['BMI'] > 0]['BMI'])
bmi = wnba_players[wnba_players['BMI'] > 0]

sns.boxplot(x='Pos', y='BMI', data=bmi)

sns.swarmplot(x='Pos', y='BMI', data=bmi, color='0')

plt.title('BMI of Players per position')

plt.xlabel('Pos')

plt.ylabel('BMI')
nba_players = pd.read_csv('../input/nba-players-stats-20142015/players_stats.csv')

nba_players['Sex'] = 'Male'

nba_players.rename(index=str, columns={'Collage': 'College'}, inplace=True)

# wnba_players.rename(index=str, columns={'15:00': '3PM'}, inplace=True)

wnba_players['Sex'] = 'Female'

all_players = pd.concat([nba_players, wnba_players])

all_players.replace(['G', 'G/F', 'F', 'F/C'], ['PG', 'SG', 'SF', 'PF'], inplace=True)

sns.violinplot(x='Pos', y='Height', hue='Sex', data=all_players, split=True)
