# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import matplotlib.pyplot as plt

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



with sqlite3.connect('../input/database.sqlite') as con:

    matches = pd.read_sql_query("SELECT * from Match", con)



df = matches[['id', 'country_id', 'league_id', 'season', 'home_team_goal', 'away_team_goal', 'PSH', 'PSD', 'PSA']]

df = df.dropna()



def set_winner(row):

    if (row.home_team_goal > row.away_team_goal):

        return 'PSH'

    elif (row.home_team_goal < row.away_team_goal):

        return 'PSA'

    return 'PSD'

df['winning_odds'] = df.apply(set_winner, axis=1)

df.head()

# Any results you write to the current directory are saved as output.
def get_average_roi():

    result = 0

    turnover = 0

    for i, row in df.iterrows():

        output = random.choice(['PSH', 'PSD', 'PSA'])

        turnover += 1

        result -= 1

        if (output == row.winning_odds):

            result += row[row.winning_odds]

    return result, turnover



result, turnover = 0, 0

for _ in range(5):

    r, t = get_average_roi()

    result += r

    turnover += t



print('Net result: {} on {} turnover, ROI: {}'.format(result, turnover, result / turnover * 100))
df['margin'] = 1 / df['PSH'] + 1 / df['PSD'] + 1 / df['PSA'] - 1

print(df['margin'].mean())
import math



def get_average_roi_dict(df):

    d = {}

    result = 0

    turnover = 0

    for i, row in df.iterrows():

        for key in ('PSH', 'PSD', 'PSA'):

            odds = row[key]

            rounded_odds = round(odds * 2) / 2

            if (not rounded_odds in d):

                d[rounded_odds] = []

            d[rounded_odds] += [row[key] - 1] if key == row.winning_odds else [-1]

    return d



d = get_average_roi_dict(df)

legends = []

plt.figure(figsize=(12,8))

for k in sorted(d.keys()):

    v = d[k]

    if (len(v) < 1000):

        continue

    legends.append('{}'.format(k))

    plt.plot(np.cumsum(v))

plt.legend(legends)

plt.show()