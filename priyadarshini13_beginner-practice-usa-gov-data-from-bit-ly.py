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


fd = "../input/usa-gov/usa_gov.json"

#Reading first line using readline()

open(fd).readline()



#Reading all lines using readlines()- restricting the  total number of returned bytes to 500:

open(fd).readlines(500)
#converting json string to dictionary object

import json



#parsing json data to python object

records = [json.loads(line) for line in open(fd,'rb')]

#printing the result - list of python dictionary

records[0]
#Accessing the timezone 'tz' of first record

records[0]['tz']
#finding most commonly occuring timezones

time_zones = [line['tz'] for line in records]

time_zones
#Getting keyError - means - some records does not have timezone('tz')

#So, check if tz is present and then print

time_zones = [line['tz'] for line in records if 'tz' in line]

time_zones
#most often occuring timezones

from collections import Counter

counts = Counter(time_zones)

counts.most_common(10)
#Counting using pandas

from pandas import DataFrame, Series

frame = DataFrame(records)

frame.info()
frame['tz'][:10]
tz_counts = frame['tz'].value_counts()

tz_counts[:10]
#Data cleaning- Filling missing values

#filling the empty values with 'Missing' word

clean_tz = frame['tz'].fillna('Missing')

#filling the key with missing value as 'Unknown'

#'Unknown':'Missing'

clean_tz[clean_tz == ''] = 'Unknown'

tz_counts = clean_tz.value_counts()

tz_counts[:10]
#Making horizontal boxplot graph

tz_counts[:10].plot(kind='barh', rot=0)
#First row-column2 - information about application

frame['a'][1]
#row0-column50-Info about browser

frame['a'][50]
#row0-column51-Info about device

frame['a'][51]
results = Series([x.split()[0] for x in frame.a.dropna()])

results[:5]
results.value_counts()[:8]
#decompose the time zones into windows and non-windows

cframe = frame[frame.a.notnull()]

os = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')

os[:5]
#grouping the data by timezones and os

tz_os_group = cframe.groupby(['tz',os])
#forming table

agg_counts = tz_os_group.size().unstack().fillna(0)

agg_counts[:10]
#selecting top overall timezones

sorted_timez = agg_counts.sum(1).argsort()

sorted_timez[:10]
#selec the rows in hat order, slice off the last 10 rows

#select using 'take'

count_subset = agg_counts.take(sorted_timez)[-10:]

print(count_subset)

count_subset.plot(kind = 'barh', stacked=True)
#To plot with relative percentage of users

#division of dataframe elements with the series object along the index axis

normed_subset = count_subset.div(count_subset.sum(1),axis=0)

print(normed_subset)

normed_subset.plot(kind='barh', stacked=True)