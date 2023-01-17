# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/football.csv')
# building data frames from scratch
country = ['Greece', 'France']
population = [12, 78]
list_label = ['country', 'population']
list_col = [country, population]
zipped = zip (list_label, list_col)
zipped_list = list(zipped)
dictionary = dict(zipped_list)
df = pd.DataFrame(dictionary)
# add new column

df ['capital'] = ['athens', 'paris']
# Broadcasting: create new column and assign a value to entire column
df ['income'] = '10 million $'
# line plot
data1 = data.loc[:, ['Christiano Ronaldo', 'Lionel Messi', 'Neymar']]
data1.plot()# line plot
data1 = data.loc[:, ['Christiano Ronaldo', 'Lionel Messi', 'Neymar']]
data1.plot()
plt.show()
# subplots
data1.plot(subplots=True)
plt.show()
# scatter plot
data1.plot(kind='scatter', x='Lionel Messi', y='Neymar')
plt.show()
# Histogram: normed=True means plot the graph normalized 
data1.plot(kind='hist', y='Lionel Messi', bins=50, range=(0,100), normed=True)
plt.show()
# histogram, cumulative and subplots
fig,axes = plt.subplots(nrows=2, ncols=1)
data1.plot(kind='hist', y='Lionel Messi', bins=50, ax=axes[0], range=(20,100))
data1.plot(kind='hist', y='Lionel Messi', bins=50, ax=axes[1], range=(20,100), cumulative=True)
plt.show()
plt.savefig('hist-cumulative.png')
# INDEXING PANDAS TIME SERIES
# convert list to datetime
data2 = data.head()
time_list = ['1990-01-26', '1990-01-27', '1990-01-28', '1991-01-23', '1991-03-04']
datetime_object = pd.to_datetime(time_list)
data2 ['date'] = datetime_object

#warning
warnings.filterwarnings('ignore')
# set index as datetime
# and obtain time series data
data2 = data2.set_index('date')
# select accoruding to date
data2.loc['1990-01-26']
# select from a staring date until an ending date
data2.loc['1990-01-26':'1990-01-28']
# RESAMPLING PANDAS TIME SERIES
# resample in terms of years: use A
data2.resample('A').mean()

# resample in terms of months: use M
data2.resample('M').mean()

# to fill these nan values, we can use interpolate function
# interpolate using first value
data2.resample('M').first().interpolate('linear')

# interpolate using mean
data2.resample('M').mean().interpolate('linear')