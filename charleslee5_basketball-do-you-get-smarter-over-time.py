# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



data = pd.read_csv("/kaggle/input/Seasons_Stats.csv") 


# Group by Player + Year, add columns instead of index

df = data.groupby(['Player','Year'], as_index = False).sum()



# Drop rows with PER <= 0

# df = df[df['PER'] > 0]



# Calculate PCT change

# df['PER_PCT_CHANGE'] = df['PER'].pct_change()



# Calculate years in leauge

yil_dict = df['Player'].value_counts().to_dict()

df['YIL'] = df['Player'].apply(lambda x: yil_dict[x])

df = df[df['YIL'] < 10]





# Find the start year of each player

start_year_dict = df.groupby('Player').min()['Year'].to_dict()



# Check whether the current row is the start year

df['start_year'] = df['Player'].apply(lambda x: start_year_dict[x])

df['start_year'] = df['start_year'] == df['Year']



# Remove the start year rows (because PCT change is not applicable)

# df = df[~df['start_year']]



# Group by years in league

YIL_df = df.groupby('YIL').mean()



# Plot the data

# plt.plot(YIL_df.index, YIL_df['PER_PCT_CHANGE'])



plt.plot(YIL_df.index, YIL_df['PER'])

plt.plot(YIL_df.index, YIL_df['VORP'])
