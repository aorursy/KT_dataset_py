# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/data.csv')

data_from = data[['from_lat', 'from_long']]

y=data_from['from_lat']

x=data_from['from_long']
sns.scatterplot(x=x, y=y)
data_from.plot.hexbin(x='from_long', y='from_lat', gridsize=8)
data_from.plot.hexbin(x='from_long', y='from_lat', gridsize=4)
d = np.array(data[['from_date']].dropna())

for i in range(len(d)):

    d[i] = datetime.strptime(d[i][0], "%m/%d/%Y %H:%M")
new = [[i for i in range(24)] for j in range(7)]

for i in d:

    new[i[0].weekday()][i[0].hour] += 1

hours = [i for i in range(24)]
hs = [0]*24

for i in range(24):

    for j in range(7):

        hs[i] += new[j][i]

    hs[i]/=7

print (hs)

plt.plot(hours, hs)
new_d = data[['from_long', 'from_lat', 'from_date']]

new_d = np.array(new_d)

bookings_wrt_days = [[], [], [], [], [], [], []]

for i in range(len(new_d)):

    cur_day = datetime.strptime(new_d[i][2], "%m/%d/%Y %H:%M").weekday()

    bookings_wrt_days[cur_day].append([new_d[i][0], new_d[i][1]])

df = []

for i in range(7):

    bookings_wrt_days[i] = np.array(bookings_wrt_days[i])

    df.append(pd.DataFrame(bookings_wrt_days[i], columns=['from_long', 'from_lat']))
x_day = []

y_day = []

for i in range(7):

    x_mon = np.array(df[i][['from_long']])

    y_mon = np.array(df[i][['from_lat']])

    x_day.append(x_mon.reshape(len(x_mon),))

    y_day.append(y_mon.reshape(len(y_mon),))
sns.scatterplot(x_day[0], y_day[0])

df[0].plot.hexbin(x='from_long', y='from_lat', gridsize=5)
sns.scatterplot(x_day[1], y_day[1])

df[1].plot.hexbin(x='from_long', y='from_lat', gridsize=5)
sns.scatterplot(x_day[2], y_day[2])

df[2].plot.hexbin(x='from_long', y='from_lat', gridsize=5)
sns.scatterplot(x_day[3], y_day[3])

df[3].plot.hexbin(x='from_long', y='from_lat', gridsize=5)
sns.scatterplot(x_day[4], y_day[4])

df[4].plot.hexbin(x='from_long', y='from_lat', gridsize=5)
sns.scatterplot(x_day[5], y_day[5])

df[5].plot.hexbin(x='from_long', y='from_lat', gridsize=5)
sns.scatterplot(x_day[6], y_day[6])

df[6].plot.hexbin(x='from_long', y='from_lat', gridsize=5)