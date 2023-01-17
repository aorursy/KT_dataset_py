# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path = "../input/international-football-results-from-1872-to-2017/results.csv"
football_data = pd.read_csv(file_path)
football_data
football_data.tournament.value_counts()
sns.jointplot(x = football_data['tournament'].value_counts().index, y = football_data['tournament'].value_counts().values)
print('Home score details\n', + football_data.home_score.describe())
print('\n\nAway score details\n', + football_data.away_score.describe())
#list(football_data.tournament.unique())
tourneys = football_data.groupby('tournament').nunique()
tourneys
football_data.info()
#Convert date timestamps to categorical features so that we can use in a predictive model
#football_data = football_data.assign(day = football_data.date.dt.day,
#                                     month = football_data.date.dt.month,
#                                     year = football_data.date.dt.year)
not_neutral_home_win = football_data.loc[(football_data.home_score > football_data.away_score) & (football_data.neutral == 0)]
neutral_home_win = football_data.loc[(football_data.home_score > football_data.away_score) & (football_data.neutral == 1)]
not_neutral_away_win = football_data.loc[(football_data.home_score < football_data.away_score) & (football_data.neutral == 0)]
neutral_away_win = football_data.loc[(football_data.home_score < football_data.away_score) & (football_data.neutral == 1)]
print('When playing at home,there are {} wins recorded out of {} games.'.format(len(not_neutral_home_win),len(football_data)))
print('This is {}%'.format(len(not_neutral_home_win)/len(football_data)*100))

print('\nWhen playing at a neutral stadium,there are {} home team wins recorded out of {} games.'.format(len(neutral_home_win),len(football_data)))
print('This is {}%'.format(len(neutral_home_win)/len(football_data)*100))

print('\nWhen playing away,there are {} wins recorded out of {} games.'.format(len(not_neutral_away_win),len(football_data)))
print('This is {}%'.format(len(not_neutral_away_win)/len(football_data)*100))

print('\nWhen playing at a neutral stadium,there are {} away team wins recorded out of {} games.'.format(len(neutral_away_win),len(football_data)))
print('This is {}%'.format(len(neutral_away_win)/len(football_data)*100))