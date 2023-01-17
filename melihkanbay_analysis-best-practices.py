# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/police/police.csv")
data.head()
data.shape
data.dtypes
data.isnull().sum()
data.drop("county_name",axis = 1,inplace = True)
data.columns
# alternative method

data.dropna(axis = "columns", how = "all")
data[data.violation == "Speeding"].driver_gender.value_counts(normalize=True)
data[data.driver_gender == "M"].violation.value_counts(normalize = True)
data[data.driver_gender == "F"].violation.value_counts(normalize = True)
data.groupby("driver_gender").violation.value_counts(normalize = True)
data.search_conducted.value_counts(normalize = True)
data.search_conducted.mean()
data.groupby("driver_gender").search_conducted.mean()
data.groupby(["violation","driver_gender"]).search_conducted.mean()
data.isnull().sum()
data.search_conducted.value_counts()
data.search_type.value_counts(dropna = False)
data["frisk"] = data.search_type.str.contains("Protective Frisk")
data.frisk.value_counts(dropna = False)
data.frisk.sum()
data.frisk.mean()
data.stop_date.str.slice(0, 4).value_counts()
combined =data.stop_date.str.cat(data.stop_time, sep = " ")

data["stop_datetime"] = pd.to_datetime(combined)
data.stop_datetime.dt.year.value_counts()
data.drugs_related_stop.dtype
# line plot by default (for a Series)

data.groupby(data.stop_datetime.dt.hour).drugs_related_stop.mean().plot()
# alternative: count drug-related stops by hour

data.groupby(data.stop_datetime.dt.hour).drugs_related_stop.sum().plot()
data.stop_datetime.dt.hour.value_counts().plot()
data.stop_datetime.dt.hour.value_counts().sort_index().plot()
data.stop_duration.value_counts(dropna = False)
data[(data.stop_duration == "1")|(data.stop_duration == "2")].stop_duration = "Nan"
# assignment statement did not work

data.stop_duration.value_counts()
# solves SettingWithCopyWarning USE .loc

data.loc[(data.stop_duration == '1') | (data.stop_duration == '2'), 'stop_duration'] = 'NaN'
# confusing

data.stop_duration.value_counts(dropna=False)
# replace 'NaN' string with actual NaN value

import numpy as np

data.loc[data.stop_duration == 'NaN', 'stop_duration'] = np.nan
data.stop_duration.value_counts(dropna=False)
# alternative method

data.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)
mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}

data['stop_minutes'] = data.stop_duration.map(mapping)
# matches value_counts for stop_duration

data.stop_minutes.value_counts()
data.groupby('violation_raw').stop_minutes.mean()
data.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])
data.groupby('violation_raw').stop_minutes.mean().plot()
data.groupby('violation_raw').stop_minutes.mean().plot(kind='bar')
data.groupby('violation_raw').stop_minutes.mean().sort_values().plot(kind='barh')
data.groupby('violation').driver_age.describe()
data.driver_age.plot(kind='hist')
data.driver_age.value_counts().sort_index().plot()
data.hist('driver_age', by='violation')

plt.show()
data.hist('driver_age', by='violation', sharex=True)

plt.show()
# this better then upside

data.hist('driver_age', by='violation', sharex=True, sharey=True)

plt.show()
data['new_age'] = data.stop_datetime.dt.year - data.driver_age_raw

data[['driver_age', 'new_age']].hist()

plt.show()
data[['driver_age', 'new_age']].describe()
data[(data.new_age < 15) | (data.new_age > 99)].shape
data.driver_age_raw.isnull().sum()
5621-5327
# driver_age_raw NOT MISSING, driver_age MISSING

data[(data.driver_age_raw.notnull()) & (data.driver_age.isnull())].head()
# set the ages outside that range as missing

data.loc[(data.new_age < 15) | (data.new_age > 99), 'new_age'] = np.nan

data.new_age.equals(data.driver_age)