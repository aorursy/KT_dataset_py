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
team_stats = pd.read_csv('/kaggle/input/college-basketball-dataset/cbb.csv')

team_stats.head(5)
team_stats[team_stats['YEAR']==2019].shape[0]
team_stats.groupby('YEAR').size()
team_stats.groupby('TEAM').size()[team_stats.groupby('TEAM').size() == 1]
avg_off = team_stats['ADJOE'].mean()

avg_def = team_stats['ADJDE'].mean()

print(avg_off,  avg_def, sep = ',')


team_stats[team_stats['POSTSEASON'] == 'Champions']['ADJOE'].mean() -avg_off

avg_def - team_stats[team_stats['POSTSEASON'] == 'Champions']['ADJDE'].mean()
team_stats['ADJOE'].idxmax()

team_stats.loc[1]['POSTSEASON']
games_Played= team_stats.groupby('G')['ADJOE'].mean()

print(games_Played)
games_Played.plot()