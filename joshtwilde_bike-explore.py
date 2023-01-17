# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-white')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read in the training data and show header

df = pd.read_csv('../input/train.csv')

df.head(10)
# Seems like the hour of the day should matter

df['hour'] = df['datetime'].map(lambda x: pd.to_datetime(x).hour)



# And maybe it matters differently for registered users (commuters?) and casual users

df.groupby('hour')[['registered','casual']].mean().plot(kind='bar', stacked=False)



# Looks like this hypothesis has merit - can clearly see the commuting hours

# TODO: what is the best way to model these shapes?
# Monthly seasonality must matter as well. Let's see how the seasons are defined here.

df['month'] = df['datetime'].map(lambda x: pd.to_datetime(x).month)

df.groupby('season').month.unique()
# Hmm, this is probably not how I want to define seasons

new_seasons = {12:'winter', 1:'winter', 2:'winter', 3:'spring', 4:'spring', 5:'spring', 6:'summer', 7:'summer', 8:'summer', 9:'fall', 10:'fall', 11:'fall'}

df['season_adj'] = df.month.map(new_seasons)



# Plotting 4 histograms on one plot is kind of messy

for season, group in df.groupby('season_adj')[['registered']]:

    group.registered.plot(kind='hist', alpha=0.4, legend=True, label=season, bins=25, normed=True)
# Instead, a KDE plot looks like it tells us more information

# Usage patterns in summer and fall look similar, though slightly higher in summer

# Spring comes in third, and winter is last as expected

for season, group in df.groupby('season_adj')[['registered']]:

    group.registered.plot(kind='kde', legend=True, label=season)
# Is it the same for casual users? Yes, makes sense

for season, group in df.groupby('season_adj')[['casual']]:

    group.registered.plot(kind='kde', legend=True, label=season)
# Can also show this with a simpler plot by just showing means

df.groupby('month')['count'].mean().plot(kind='bar')

# TODO: Include some kind of seasonality, probably the 4 seasons are fine
# Whether or not it is a working day should matter

# And it matters for registered users

df.hist(by='workingday', column='registered', sharex=True, sharey=True, normed=True)
# What about for casual users?

# Actually the direction is opposite - more casual users on non-working days, this makes sense

df.hist(by='workingday', column='casual', sharex=True, sharey=True, normed=True)



# TODO: Does seem like there need to be two different models, one for casual users and one for registered
# How many times do we get each weather item?

# Only once for the heavy rain! This means it should be excluded somehow from weather calcs

df.groupby('weather').datetime.count()
# Looking at weather, hypothesis is that as weather variable increases, usage decreases

# And this impact should be the same for both registered and casual users

# Looks like the biggest impact is between clear and any precip, which makes sense



for weather, group in df.groupby('weather')[['count']]:

    if weather != 4: group['count'].plot(kind='kde', legend=True, label=weather, title='All Seasons')
# Temperature definitely matters, "feels like" should matter more

# Effect might differ between the registered and casual users



# First, lets try to look at the temperature relationship with users in straight scatter plot

df.plot(x='atemp', y='count', kind='scatter')
# Hmm, looks like there is a lot of noise in that calculation

# Let's try it only using registered users

df.plot(x='atemp', y='registered', kind='scatter')
# This doesn't help much...maybe only at commuting hours?

commute = [7,8,9,16,17,18,19]



df[df.hour.isin(commute)].plot(x='atemp', y='registered', kind='scatter')
# Really hard to tell if there is a relationship here from the scatter plots

# Let's try another visualization method, this time with temperature quantiles

# Let's start with deciles



# Wow looks like temperature doesnt matter in the summer during commuting hours! Guess this does make sense

df['atemp_deciles'] = pd.qcut(df.atemp, q=10, labels=False)

df[(df.season_adj == 'summer') & (df['workingday'] == 1) & (df.hour.isin(commute))].boxplot(column='registered', by='atemp_deciles')