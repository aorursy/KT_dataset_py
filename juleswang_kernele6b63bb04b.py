# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv',  parse_dates = True, index_col='datetime',
                    dtype = {
                        'weather': 'category',
                   })
train.head()
train['hour'] = train.index.hour
train['month'] = train.index.month
train['weekday'] = train.index.weekday
hourAggregated = pd.pivot_table(train, index=['hour'], values=['count'], columns=['workingday'])
hourAggregated.columns = ['non-workingday', 'workingday']
hourAggregated.plot(grid=True, style='-s', figsize=(12, 6))
registeredAggregated = pd.pivot_table(train, index=['hour'], values=['casual', 'registered'],columns=['workingday'])
registeredAggregated.columns = ['casual in non-workingdays', 'casual in workingdays', 'registered in non-workingdays', 'registered in workingdays']
registeredAggregated.plot(grid=True, style='-s', figsize=(12, 6), title="casual is more in non-workingdays, much less in workingdays ")
train['new_season'] = np.floor((train.month - 3) % 12 / 3)
seasonAggregated = pd.pivot_table(train, index=['hour'], values=['count'], columns=['new_season'])
seasonAggregated.columns = ['spring', 'summer', 'fall', 'winter']
seasonAggregated.plot(grid=True, style='-s', figsize=(12, 6), title="People rent bikes more in summer, much less in winter")
seasontempAggregated = pd.pivot_table(train, index=['hour'], values=['temp'], columns=['new_season'])
seasontempAggregated.columns = ['spring', 'summer', 'fall', 'winter']
seasontempAggregated.plot(grid=True, style='-s', figsize=(12, 6), title="Obviously, temp is low in winter, high in summer")
weatherAggregated = pd.pivot_table(train, index=['hour'], values=['count'], columns=['weather'])
weatherAggregated.columns = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain']
weatherAggregated.plot(grid=True, style='-s', figsize=(12, 6), title="People rent bikes more when weather is good")
# temp - temperature in Celsius
tempAggregated = pd.pivot_table(train, index=['temp'], values=['count'])
tempAggregated.plot(grid=True, style='-s', figsize=(12, 6), title="best temp for renting bikes is about 32C ")
#atemp - "feels like" temperature in Celsius
atempAggregated = pd.pivot_table(train, index=['atemp'], values=['count'])
atempAggregated.plot(grid=True, style='-s', figsize=(12, 6), title="best atemp for renting bikes is about 40C ")
humidityAggregated = pd.pivot_table(train, index=['humidity'], values=['count'])
humidityAggregated.plot(grid=True, style='-s', figsize=(12, 6), title="less people rents bikes when humidity is high")
windAggregated = pd.pivot_table(train, index=['windspeed'], values=['count'])
windAggregated.plot(grid=True, style='-s', figsize=(12, 6), title="windspeed")