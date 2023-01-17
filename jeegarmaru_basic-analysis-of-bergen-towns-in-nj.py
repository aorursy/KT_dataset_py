# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', 30)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-new-jersey-nj-local-dataset/Covid-19-NJ-Bergen-Municipality.csv', index_col='Date', parse_dates=['Date'])

df
# Just fixing a bad data issue/typo

df.loc['2020-04-23', 'Elmwood Park'] = 407
towns = set(df.columns) - set(['Total Presumptive Positives', 'New Daily cases'])

print("A few towns :")

list(towns)[:10]
highest_towns = df.iloc[-1][towns].nlargest(10).index

#Also, adding some towns that are of interest to me

print("Towns with highest number of cases + some towns of interest to me:")

highest_towns = list(highest_towns) + ['Edgewater', 'Fort Lee']

highest_towns
ax = df[highest_towns].plot(figsize=(20, 10), rot=45)

ax.set_ylabel('Number of positive cases')

ax.set_title('Total Positive cases over time by towns', fontweight='bold', fontsize='x-large')

ax.set_xticks(df[highest_towns].index)

ax.set_xticklabels(df[highest_towns].index.strftime('%b-%d'))

plt.show()
# Skipping initial numbers to focus more on the most recent trends/numbers

pct_change = df[40:].pct_change().rolling(10, 1).mean() * 100.0

ax = pct_change[highest_towns].plot(figsize=(20, 10), rot=45)

ax.set_ylabel('Percent daily rate of change')

ax.set_title('Percent daily rate of change over time by towns', fontweight='bold', fontsize='x-large')

ax.set_xticks(pct_change.index)

ax.set_xticklabels(pct_change.index.strftime('%b-%d'))

plt.show()
# We'll look at data since March 20th to avoid the erratic data before.

# Also, we'll include only 10 towns that have the highest number of cases

recent_df = df[df.index > '2020-03-20']

recent_pct_change = recent_df.pct_change()

recent_pct_change[highest_towns].mean().sort_values()
# We'll take a Simple Moving Average to smoothen the irregularities with the raw numbers

days_to_double = 100.0/pct_change[15:]

ax = days_to_double[set(highest_towns) - {'Elmwood Park', 'Teaneck'}].rolling(5, 1).mean().plot(figsize=(20, 10), rot=45)

ax.set_ylabel('Number of days for cases to double')

ax.set_title('Number of days for cases to double over time by towns', fontweight='bold', fontsize='x-large')

ax.set_xticks(days_to_double.index)

ax.set_xticklabels(days_to_double.index.strftime('%b-%d'))

plt.show()
melted_df = df[highest_towns].reset_index().melt(id_vars='Date', var_name='Town', value_name='Total Cases')

melted_df
melted_df['New Cases'] = melted_df.groupby('Town')['Total Cases'].transform(lambda x: x.diff())

melted_df
melted_df['New Cases'] = melted_df.groupby('Town')['New Cases'].transform(lambda x: x.rolling(8, 1).mean())
plt.figure(figsize=(20,20))

grid = sns.lineplot(x="Total Cases", y="New Cases", hue="Town", data=melted_df)

grid.set(xscale="log", yscale="log")

grid.set_title('New cases vs existing cases at log scale', fontweight='bold', fontsize='x-large')

plt.show()
towns = ['Cliffside Park', 'Edgewater', 'Englewood', 'Englewood Cliffs', 'Fort Lee', 'Hackensack', 'Leonia', 'Paramus', 'Palisades Park']

int_df = df[-15:][towns]
ax = int_df.plot(figsize=(20, 10), rot=45)

ax.set_ylabel('Number of positive cases')

ax.set_title('Total Positive cases over time by towns', fontweight='bold', fontsize='x-large')

ax.set_xticks(int_df[towns].index)

ax.set_xticklabels(int_df[towns].index.strftime('%b-%d'))

plt.show()
# Skipping initial numbers to focus more on the most recent trends/numbers

pct_change = int_df.pct_change().rolling(4, 1).mean() * 100.0

ax = pct_change.plot(figsize=(20, 10), rot=45)

ax.set_ylabel('Percent daily rate of change')

ax.set_title('Percent daily rate of change over time by towns', fontweight='bold', fontsize='x-large')

ax.set_xticks(pct_change.index)

ax.set_xticklabels(pct_change.index.strftime('%b-%d'))

plt.show()
# We'll take a Simple Moving Average to smoothen the irregularities with the raw numbers

days_to_double = 100.0/pct_change

ax = days_to_double.rolling(5, 1).mean().plot(figsize=(20, 10), rot=45)

ax.set_ylabel('Number of days for cases to double')

ax.set_title('Number of days for cases to double over time by towns', fontweight='bold', fontsize='x-large')

ax.set_xticks(days_to_double.index)

ax.set_xticklabels(days_to_double.index.strftime('%b-%d'))

plt.show()