# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import dateutil


# Any results you write to the current directory are saved as output.
# Load data from csv file
data = pd.read_csv('../input/phone_data.csv')
data.head()
# Convert date from string to date times
data['date'] = data['date'].apply(dateutil.parser.parse, dayfirst=True)
data.head()
# How many rows the dataset
data['item'].count()

#another form
data.info()
# What was the longest phone call / data entry?
data['duration'].max()
# How many seconds of phone calls are recorded in total?
data['duration'][data['item'] == 'call'].sum()
# How many entries are there for each month?
data['month'].value_counts()
# Number of non-null unique network entries
data['network'].nunique()
data.groupby(['month']).groups.keys()
data.groupby(['item']).groups.keys()
data.groupby(['network']).groups.keys()
data.groupby(['network_type']).groups.keys()
len(data.groupby(['month']).groups['2014-11'])
len(data.groupby(['month']).groups['2015-02'])
# Get the first entry for each month
data.groupby('month').first()
# Get the sum of the durations per month
data.groupby('month')['duration'].sum()
# Get the number of dates / entries in each month
data.groupby('month')['date'].count()
# What is the sum of durations, for calls only, to each network
data[data['item'] == 'call'].groupby('network')['duration'].sum()
# How many calls, sms, and data entries are in each month?
data.groupby(['month', 'item'])['date'].count()
# How many calls, texts, and data are sent per month, split by network_type?
data.groupby(['month', 'network_type'])['date'].count()

data.groupby('month')['duration'].sum() 
# produces Pandas Series
data.groupby('month')[['duration']].sum() 
# Produces Pandas DataFrame
data.groupby('month', as_index=False).agg({"duration": "sum"})
# Group the data frame by month and item and extract a number of stats from each group
data.groupby(['month', 'item']).agg({'duration':sum,      # find the sum of the durations for each group
                                     'network_type': "count", # find the number of network type entries
                                     'date': 'first'})    # get the first date per group
# Group the data frame by month and item and extract a number of stats from each group
data.groupby(['month', 'item']).agg({'duration': [min, max, sum],      # find the min, max, and sum of the duration column
                                     'network_type': "count", # find the number of network type entries
                                     'date': [min, 'first', 'nunique']})    # get the min, first, and number of unique dates per group