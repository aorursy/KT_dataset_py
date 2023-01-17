# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

%matplotlib inline
free_throw_all = pd.read_csv('../input/free_throws.csv')

free_throw_all.tail()
free_throw_all.head()
free_throw_player = free_throw_all[['season', 'player', 'shot_made']]

free_throw_player.head()
overview = pd.pivot_table(free_throw_player,

               values='shot_made',

               index=['season', 'player'],

               aggfunc=[np.sum, len, np.mean])

overview.columns = ['score', 'total_shot', 'hit rate']
noise = overview.total_shot < 80

noise_ind = overview[noise].index

overview = overview.drop(noise_ind)
overview.sort_values('hit rate', ascending=False).iloc[:10]
from datetime import datetime

ptime = lambda x: datetime.time(datetime.strptime(x, '%M:%S'))
free_throw_all['time'] = free_throw_all.time.map(ptime)
free_throw_all.tail()
def score_to_score(x):

    piece = [int(y.strip()) for y in x.split('-')]

    if abs(piece[0] - piece[1]) < 10:

        return True

    else:

        return False
free_throw_all['tight_game'] = free_throw_all.end_result.map(score_to_score)

free_throw_all.tail()
from datetime import time

last_two_minutes = time(hour=0, minute=2, second=0)

last_quater = free_throw_all.period == 4.0

last_2min = free_throw_all.time < last_two_minutes
free_throw_all.where(last_quater).where(last_2min).where(free_throw_all.tight_game).dropna()
count1 = lambda x: sum(1 for y in x)

regular = free_throw_all.playoffs == 'regular'

regular_d = free_throw_all[regular]

regular_D = regular_d.pivot_table(values='shot_made',

                      index=['season', 'player'],

                      aggfunc=[np.sum, len, np.mean])

regular_D.columns = ['score', 'totalshot', 'hit_rate']
playoffs = free_throw_all.playoffs == 'playoffs'

playoffs_d = free_throw_all[playoffs]

playoffs_D = playoffs_d.pivot_table(values='shot_made',

                      index=['season', 'player'],

                      aggfunc=[np.sum, len, np.mean])

playoffs_D.columns = ['score', 'totalshot', 'hit_rate']
reg_vs_poff = pd.merge(regular_D, playoffs_D, left_index=True, right_index=True, suffixes=('_regular', '_playoffs'))
noise = reg_vs_poff.totalshot_playoffs < 5

noise_ind = reg_vs_poff[noise].index

reg_vs_poff = reg_vs_poff.drop(noise_ind)
reg_vs_poff
reg_vs_poff.sort_values(['totalshot_regular', 'hit_rate_regular'], ascending=False).iloc[:11]
reg_vs_poff.sort_values(['totalshot_playoffs', 'hit_rate_playoffs'], ascending=False).iloc[:11]
reg_vs_poff[['hit_rate_playoffs']].plot(kind='hist')
AllStar_2016 = DataFrame({'player': ['Dwyane Wade', 'Kyle Lowry',

                          'LeBron James', 'Paul George', 'Carmelo Anthony',

                          'Stephen Curry', 'Russell Westbrook', 'Kobe Bryant',

                          'Kevin Durant', 'Kawhi Leonard'],

                          'year': [2003, 2006, 2003, 2010, 2003, 2009, 2008,

                          1996, 2007, 2011],

                          'position': ['G', 'G', 'F', 'F', 'F', 'G', 'G',

                                       'G', 'F', 'F']})
def AllStar(x):

    b = x in AllStar_2016['player'].values

    return b



list = free_throw_all['player'].map(AllStar)

AllStar_16 = free_throw_all[list]
AllStar_2016 = pd.merge(AllStar_16, AllStar_2016,

                        left_on='player', right_on='player',

                        how='inner')

AllStar_2016.tail()
Year = lambda x: int(x[-4:])

AllStar_2016['Year'] = AllStar_2016['season'].map(Year)
AllStar_2016['career'] = AllStar_2016['Year'] - AllStar_2016['year']
table = AllStar_2016.pivot_table(index=['player', 'Year'],

                         values='shot_made',

                         aggfunc=np.mean)
table = table.unstack()
table = table.T
fig, axes = plt.subplots(5, 2, sharey=True, sharex=True)

index = 0

for i in range(5):

    for j in range(2):

        table.iloc[:, index].plot(ax=axes[i, j],

                            figsize=(20, 20),

                            ylim=[0.6, 1],

                            legend=False,

                            yticks=[0.6, 0.8, 1],

                            title=str(table.columns[index]))

        index += 1
free_throw_all.pivot_table(index='period', values='shot_made', aggfunc=np.mean)