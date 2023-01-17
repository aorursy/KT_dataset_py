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
# ri stands for Rhode Island

ri = pd.read_csv('../input/my-new-dataset/police.csv')
# what does each row represent?

ri.head()
# what do these numbers mean?

ri.shape
# what do these types mean?

ri.dtypes
# what are these counts? how does this work?

ri.isnull().sum()
# axis=1 also works, inplace is False by default, inplace=True avoids assignment statement

ri.drop('county_name', axis='columns', inplace=True)
ri.shape
ri.columns
# Alternative method

#  ri.dropna(axis='columns', how='all')
# when someone is stopped for speeding, how often is it a man or woman?

ri[ri.violation == 'Speeding'].driver_gender.value_counts(normalize=True)
# Alternative method

ri.loc[ri.violation == 'Speeding', 'driver_gender'].value_counts(normalize=True)
# when a man is pulled over, how often is it for speeding?

ri[ri.driver_gender == 'M'].violation.value_counts(normalize=True)
# repeat for women

ri[ri.driver_gender == 'F'].violation.value_counts(normalize=True)
# combines the two lines above

ri.groupby('driver_gender').violation.value_counts(normalize=True)
# ignore gender for the moment

ri.search_conducted.value_counts(normalize=True)
# how does this work?

ri.search_conducted.mean()
# search rate by gender

ri.groupby('driver_gender').search_conducted.mean()
# include a second factor

ri.groupby(['violation', 'driver_gender']).search_conducted.mean()
ri.isnull().sum()
# maybe search_type is missing any time search_conducted is False?

ri.search_conducted.value_counts()
# test that theory, why is the Series empty?

ri[ri.search_conducted == False].search_type.value_counts()
# value_counts ignores missing values by default

ri[ri.search_conducted == False].search_type.value_counts(dropna=False)
# when search_conducted is True, search_type is never missing

ri[ri.search_conducted == True].search_type.value_counts(dropna=False)
# alternative method

ri[ri.search_conducted == True].search_type.isnull().sum()
# multiple types are separated by commas

ri.search_type.value_counts(dropna=False)
# use bracket notation when creating a column

ri['frisk'] = ri.search_type == 'Protective Frisk'
ri.frisk.dtype
# includes exact matches only

ri.frisk.sum()
# is this the answer?

ri.frisk.mean()
# uses the wrong denominator (includes stops that didn't involve a search)

ri.frisk.value_counts()
# includes partial matches

ri['frisk'] = ri.search_type.str.contains('Protective Frisk')
# seems about right

ri.frisk.sum()
# frisk rate during a search

ri.frisk.mean()
# str.contains preserved missing values from search_type

ri.frisk.value_counts(dropna=False)
# this works, but there's a better way

ri.stop_date.str.slice(0, 4).value_counts()
# make sure you create this column

combined = ri.stop_date.str.cat(ri.stop_time, sep=' ')

ri['stop_datetime'] = pd.to_datetime(combined)
ri.dtypes
# why is 2005 so much smaller?

ri.stop_datetime.dt.year.value_counts()
ri.drugs_related_stop.dtype
# baseline rate

ri.drugs_related_stop.mean()
# can't groupby 'hour' unless you create it as a column

ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean()
# line plot by default (for a Series)

ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.mean().plot()
# alternative: count drug-related stops by hour

ri.groupby(ri.stop_datetime.dt.hour).drugs_related_stop.sum().plot()
ri.stop_datetime.dt.hour.value_counts()
ri.stop_datetime.dt.hour.value_counts().plot()
ri.stop_datetime.dt.hour.value_counts().sort_index().plot()
# alternative method

ri.groupby(ri.stop_datetime.dt.hour).stop_date.count().plot()
# mark bad data as missing

ri.stop_duration.value_counts()
# what two things are still wrong with this code?

ri[(ri.stop_duration == '1') | (ri.stop_duration == '2')].stop_duration = 'NaN'
# assignment statement did not work

ri.stop_duration.value_counts()
# solves Setting With Copy Warning

ri.loc[(ri.stop_duration == '1') | (ri.stop_duration == '2'), 'stop_duration'] = 'NaN'
# confusing!

ri.stop_duration.value_counts(dropna=False)
# replace 'NaN' string with actual NaN value

import numpy as np

ri.loc[ri.stop_duration == 'NaN', 'stop_duration'] = np.nan
ri.stop_duration.value_counts(dropna=False)
# alternative method

#ri.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)
# make sure you create this column

mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}

ri['stop_minutes'] = ri.stop_duration.map(mapping)
# matches value_counts for stop_duration

ri.stop_minutes.value_counts()
ri.groupby('violation_raw').stop_minutes.mean()
ri.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])
# what's wrong with this?

ri.groupby('violation_raw').stop_minutes.mean().plot()
# how could this be made better?

ri.groupby('violation_raw').stop_minutes.mean().plot(kind='bar')
ri.groupby('violation_raw').stop_minutes.mean().sort_values().plot(kind='barh')
# good first step

ri.groupby('violation').driver_age.describe()
# histograms are excellent for displaying distributions

ri.driver_age.plot(kind='hist')
# similar to a histogram

ri.driver_age.value_counts().sort_index().plot()
# can't use the plot method

ri.hist('driver_age', by='violation')
# what changed? how is this better or worse?

ri.hist('driver_age', by='violation', sharex=True)
# what changed? how is this better or worse?

ri.hist('driver_age', by='violation', sharex=True, sharey=True)
ri.head()
# appears to be year of stop_date minus driver_age_raw

ri.tail()
ri['new_age'] = ri.stop_datetime.dt.year - ri.driver_age_raw
# compare the distributions

ri[['driver_age', 'new_age']].hist()
# compare the summary statistics (focus on min and max)

ri[['driver_age', 'new_age']].describe()
# calculate how many ages are outside that range

ri[(ri.new_age < 15) | (ri.new_age > 99)].shape
# raw data given to the researchers

ri.driver_age_raw.isnull().sum()
# age computed by the researchers (has more missing values)

ri.driver_age.isnull().sum()
# what does this tell us? researchers set driver_age as missing if less than 15 or more than 99

5621-5327
# driver_age_raw NOT MISSING, driver_age MISSING

ri[(ri.driver_age_raw.notnull()) & (ri.driver_age.isnull())].head()
# set the ages outside that range as missing

ri.loc[(ri.new_age < 15) | (ri.new_age > 99), 'new_age'] = np.nan
ri.new_age.equals(ri.driver_age)