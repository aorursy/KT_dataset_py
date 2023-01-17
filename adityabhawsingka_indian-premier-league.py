#import modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('ggplot')

# increase default figure and font sizes for easier viewing

plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['font.size'] = 14
#read csv into a pandas dataframe.

matches = pd.read_csv("../input/matches.csv")
#check the first 5 rows.

matches.head()
matches.isnull().sum()
matches[matches.city.isnull() == True]
matches[matches.venue == 'Dubai International Cricket Stadium'].city.value_counts(dropna=False)
#Update city as 'Dubai' wherever it is null.

matches['city'].fillna(value='Dubai',inplace=True)
matches[matches.winner.isnull() == True]
#Update winner and player_of_match columns as 'no result' wherever they are null.

matches['winner'].fillna(value='no result',inplace=True)

matches['player_of_match'].fillna(value='no result',inplace=True)
matches[matches.umpire1.isnull() == True]
#Update umpire1 and umpire2 as 'not available' wherever their value is null.

matches['umpire1'].fillna(value='not available',inplace=True)

matches['umpire2'].fillna(value='not available',inplace=True)
#drop the null column

matches.dropna(axis=1,inplace=True)
matches.shape
matches.player_of_match.value_counts().head(10).sort_values().plot(kind='barh')
vc = matches.groupby('season').player_of_match.value_counts()

imax = vc.idxmax()

print('\n\nYear       :', imax[0], '\nPlayer     :',imax[1], '\nNo. of PoM :', vc[vc.idxmax()])
matches[matches.player_of_match == 'CH Gayle'].city.value_counts().plot(kind='barh')
matches[matches.player_of_match == 'CH Gayle'].winner.value_counts().plot(kind='barh')
a = matches[(matches.player_of_match == 'CH Gayle') & (matches.team1 == 'Royal Challengers Bangalore')].team2

b = matches[(matches.player_of_match == 'CH Gayle') & (matches.team2 == 'Royal Challengers Bangalore')].team1

a.append(b).value_counts().plot(kind='barh')
matches.season.value_counts().sort_index().plot(kind='bar')
matches[matches.win_by_wickets > 0].win_by_wickets.plot.hist(bins=5)

plt.xlabel('Win by wickets')
matches[['winner','win_by_wickets']][matches.win_by_wickets > 0].boxplot(vert=False, by='winner')
matches[matches.win_by_runs > 0].win_by_runs.plot.hist(bins=15)

plt.xlabel('Win by runs')
matches[['winner','win_by_runs']][matches.win_by_runs > 0].boxplot(vert=False, by='winner')