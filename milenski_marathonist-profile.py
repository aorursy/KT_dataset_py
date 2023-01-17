import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import re



data = pd.read_csv('../input/marathon_results_2016.csv')

data = data.drop('Name', axis = 1)

list(data)
#transform a string to time in seconds

def time_to_seconds(time_str):

    try:

        h, m, s = tuple(map(int, re.split(r'[:-]', (time_str))))

        return float(h * 3600 + m*60 + s)

    except ValueError:

        return np.nan



plt.figure(figsize=(12,6))

#data.loc[:, '5K' : 'Official Time'] = data.loc[:, '5K' : 'Official Time'].apply(np.vectorize(time_to_seconds))

plt.plot(data['Pace'].dropna())
plt.plot(data['Pace'].dropna().iloc[:2000])
pace_by_country = data.groupby('Country')['Pace'].mean().sort_values()



top = pace_by_country[:6]

btm = pace_by_country[-6:]



x_axis = np.arange(len(top) + len(btm))

plt.figure(figsize=(12,6))

plt.xticks(x_axis, top.append(btm).index, rotation='vertical')

plt.bar(x_axis, top.append(btm))
tot_part = data.groupby('M/F')['Pace'].count()

plt.xticks(np.arange(2), tot_part.index)

plt.bar(np.arange(2), tot_part, color = ['pink', 'c'])
by_gender = data.groupby('M/F').mean()

plt.ylim(400, 600)

plt.xticks(np.arange(2), by_gender['Pace'].index)

plt.bar(np.arange(2), by_gender['Pace'], color = ['pink', 'c'])