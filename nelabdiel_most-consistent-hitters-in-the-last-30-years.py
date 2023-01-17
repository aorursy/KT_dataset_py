# Things we'll need.

import pandas as pd

import numpy as np

import sqlite3 as sql

%matplotlib inline

import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 15, 5
# Defining our connector the the database

conn = sql.connect('../input/database.sqlite')

c = conn.cursor()
PlayerInfo = pd.DataFrame(list(c.execute('''select player_id, name_first, 

                                             name_last, debut, final_game from player''')))

names = list(map(lambda x: x[0], c.description))

PlayerInfo.columns = names

PlayerInfo.head()
PBY = pd.DataFrame(list(c.execute('''select player_id, year, 

                                     1.*h/(ab-ibb-hbp) as batting_percentage 

                                     from batting group by year, player_id''')))

names = list(map(lambda x: x[0], c.description))

PBY.columns = names

PBY.head()
Career = PBY.groupby('player_id')['batting_percentage'].agg([np.mean, np.std]).reset_index()

Career.columns = [['player_id', 'career_batting_percentage', 'career_batting_std']]

Career.head()
YearsActive = PBY.groupby('player_id')['year'].count().reset_index()

YearsActive.columns = ['player_id', 'years_active']

YearsActive = YearsActive[YearsActive['years_active'] >= 4]
YearsActive.sort_values('years_active', 

                        ascending=False).head(10).merge(PlayerInfo, 

                                                        how='left', on='player_id')
GreatYears = PBY[PBY['batting_percentage'] >= .3].groupby('player_id')['batting_percentage'].count().reset_index()

GreatYears.columns = ['player_id', 'great_years_amount']

GreatYears.sort_values('great_years_amount', 

                       ascending=False).head(3).merge(PlayerInfo, 

                                                      how='left', on='player_id')
Stats = GreatYears.merge(YearsActive, how='left', on='player_id')



# New column of percent of years with more than .3 average

Stats['percent_of_greatness'] = Stats['great_years_amount']/Stats['years_active']



Stats = Stats.merge(Career, how='left', on='player_id')



data = Stats.merge(PlayerInfo, how='left', on='player_id')



data = data[['name_first', 'name_last','debut', 'percent_of_greatness',

             'final_game','great_years_amount', 'years_active',

             'career_batting_percentage', 'career_batting_std']]
data = data[data['debut'] >= '1984-01-01'].sort_values('percent_of_greatness', 

                                                       ascending=False)

data[:15]
np.nanpercentile(np.array(data['percent_of_greatness']), 99)
np.nanpercentile(np.array(data['percent_of_greatness']), 99.9)
# Let's graph a few

rcParams['figure.figsize'] = 9, 5

x = np.array(list(range(70)))

plt.xticks(x, np.array(data['name_last'][:70]), rotation='vertical')

#plt.plot(x, y)

plt.bar(x, np.array(data['percent_of_greatness'][:70]), color = 'red' )

plt.show()