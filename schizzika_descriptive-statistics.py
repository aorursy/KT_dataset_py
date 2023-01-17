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
import math

import numpy as np

import pandas as pd

from scipy import stats

import os
match_data = pd.read_csv("/kaggle/input/ipldata/matches.csv")

delivery_data = pd.read_csv("/kaggle/input/ipldata/deliveries.csv")
len(match_data)
match_data.head()
win_by_runs_data = match_data[match_data['win_by_runs'] > 0].win_by_runs

print(f'Number of rows with matches won by margin = {len(win_by_runs_data)}')

win_by_runs_data.head()
#Finding the Arithmetic mean of the margin wins

print(f'Arithmetic mean of margin wins = {win_by_runs_data.mean()}')
#Find the median

print(f'Median = {win_by_runs_data.median()}')
#Find the mode

print(f'Mode = {win_by_runs_data.mode()}')
#Find the range

print("Max margin = ", max(win_by_runs_data))

print("Min margin = ", min(win_by_runs_data))

print(f'Range of margin wins = {max(win_by_runs_data) - min(win_by_runs_data)}')
#Different quartiles

win_by_runs_25_perc = stats.scoreatpercentile(win_by_runs_data, 25)

win_by_runs_75_perc = stats.scoreatpercentile(win_by_runs_data, 75)



win_by_runs_iqr = stats.iqr(win_by_runs_data)

print(f'First quartile i.e (25th percentile) = {win_by_runs_25_perc}')

print(f'Second quartile i.e (50th percentile) = {win_by_runs_data.median()}')

print(f'Third quartile i.e (75th percentile) = {win_by_runs_75_perc}')

print(f'Inter Quartile Range = Q3 - Q1 = {win_by_runs_75_perc} - {win_by_runs_25_perc} = {win_by_runs_iqr}')
win_by_runs_95_perc = stats.scoreatpercentile(win_by_runs_data, 95)

print(f'95th percentile = {win_by_runs_95_perc}')
win_by_wickets_data = match_data[match_data.win_by_wickets > 0].win_by_wickets

win_by_wickets_data.head()
win_by_wickets_data.describe()
#Mean absolute deviation

print(f'Mean absolute deviation = {win_by_runs_data.mad()}')
import matplotlib.pyplot as plt
plt.boxplot(win_by_runs_data)

plt.xlabel("Margin win by runs", size = 10, color = 'black')

plt.ylabel("Runs", size = 10, color = 'black')

plt.title('Box plot of margin runs')

plt.yticks(np.arange(0, 100, step = 10))