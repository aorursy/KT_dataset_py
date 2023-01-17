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
df = pd.read_csv('../input/odi-cricket-matches-19712017/originalDataset.csv')

df.head()
df.tail()
df['Winner'].value_counts().keys()[0] # Most number of wins by a team from 1971 to 2017
df
df_india_australia = df[(df['Team 1'].str.contains('Australia') | df['Team 1'].str.contains('India'))  & (df['Team 2'].str.contains('India') | df['Team 2'].str.contains('Australia'))]
df_india_australia
df_india_australia['Winner'].value_counts()
df_in_aus_matches_played_in_pune = df_india_australia[df_india_australia['Ground'] == 'Pune']

df_in_aus_matches_played_in_pune
df_most_number_of_matches_played_at_a_ground = df_india_australia['Ground'].value_counts()

df_most_number_of_matches_played_at_a_ground.keys()[0]
df_india_australia.shape[0] # Total Number of matches palyed between India and Australia 
df_india_australia
df_west_indies = df[df['Team 1'].str.contains('West Indies') | df['Team 2'].str.contains('West Indies')]
total_number_of_matches = df_west_indies.shape

total = total_number_of_matches[0]

total
most_matches_won = df_west_indies['Winner'].value_counts(sort=True).keys()[0]

most_matches_won
total_won_number = df_west_indies['Winner'].value_counts(sort=True)[0]

total_won_number
win_ratio = int(total_won_number) / int(total)

final_win_ratio = win_ratio * 100

final_win_ratio