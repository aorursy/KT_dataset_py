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
dataframe = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
print("Total Rows:", dataframe.shape[0])

print("Total Columns", dataframe.shape[1])
dataframe.head()
dataframe['date'] = pd.to_datetime(dataframe['date'])
dataframe['date'] = pd.DatetimeIndex(dataframe['date'])
year = pd.DatetimeIndex(dataframe['date']).year.value_counts()

year = year.rename_axis("Year")

year.sort_index().plot(kind='bar', title = "Number of Gun violence by Year", figsize=(15,10))

# dataframe['month'] = pd.DatetimeIndex(dataframe['date']).month

# dataframe['day'] = pd.DatetimeIndex(dataframe['date']).day
MonthDF = pd.DataFrame({'Count':pd.DatetimeIndex(dataframe['date']).month_name().value_counts()})

# monthDF = MonthDF.set_index('Months')

MonthDF.index.name= "Month"

MonthDF.dtypes

# MonthDF.set_index =['Jan','Feb', 'March', 'Apr','May', 'Jun','Jul','Aug','Sep','Oct','Nov','Dec']

MonthDF.plot(kind='bar', title = "Number of Gun violence by Month", figsize=(15,10))
year = pd.DatetimeIndex(dataframe['date']).day.value_counts()

year = year.rename_axis("Day")

year.sort_index().plot(kind='line', title = "Number of Gun violence by Day of Month", figsize=(18,8))
dataframe.head()
dataframe.groupby('state').agg('sum')['n_killed'].sort_values().plot(kind='barh', figsize = (15,15), title="Number of Deaths in every states")
df = dataframe.groupby('state').agg('sum')[['n_killed','n_injured']]

df.plot(kind='bar', rot=0, figsize = (40,14), title="Gun Casualty by state")
dataframe = dataframe.drop(['source_url','incident_url','incident_url_fields_missing', 'incident_id'],axis=1)

dataframe.head()
dataframe['year'] = pd.DatetimeIndex(dataframe['date']).year

dataframe['month'] = pd.DatetimeIndex(dataframe['date']).month_name()

dataframe['day'] = pd.DatetimeIndex(dataframe['date']).day

dataframe.head()
df = dataframe.groupby('year')[['n_killed','n_injured']].agg('sum')

df = df.rename(index=str, columns={"n_killed":"People Killed",'n_injured':'People Injured'})

# df.head()

df.plot(kind='bar', rot=0, title="Number of Casaulties by Year", figsize=(15,10))
df = dataframe.groupby('month')[['n_killed','n_injured']].agg('sum')

df = df.rename(index=str, columns={"n_killed":"People Killed",'n_injured':'People Injured'})

# df.head()

df.sort_values(by='People Killed').plot(kind='bar', rot=0, title="Number of Casaulties by Month", figsize=(15,12))
df = dataframe.groupby('day')[['n_killed']].agg('sum')

df = df.rename(index=str, columns={"n_killed":"People Killed"})

# df.sort_index()

df.sort_values(by='People Killed', ascending = True).plot(kind='barh', title="Number of deaths by day of the month", figsize=(15,10))
dataframe['monthandday'] = dataframe['month'].map(str) + " "+ dataframe['day'].map(str)

dataframe.head()
baddaydf = pd.DataFrame(dataframe['monthandday'].value_counts())

baddaydf = baddaydf.rename(index=str, columns={'monthandday' :'Date'})

# baddaydf = baddaydf.sort_values()

baddaydf[0:10].plot(kind ='bar', title="Top 10 Days with maximum number of Gun Violence")
df = dataframe.groupby('monthandday')[['n_killed']].agg('sum')

df = df.rename(index=str, columns={"n_killed":"People Killed"})

df = df.sort_values('People Killed', ascending = False)

# df.head()

df[0:10].plot(kind='barh', title="Top 10 days of month by Number of deaths", figsize=(15,10))
dataframe = dataframe.drop(['year','month','day','monthandday'], axis=1)
dataframe['day of the week'] = pd.DatetimeIndex(dataframe['date']).day_name()

dataframe.head()
df = dataframe.groupby('day of the week').agg('sum')[['n_killed','n_injured']]

df = df.rename(index = str, columns={"n_killed":"People Killed", "n_injured":"People Injured"})

df.plot(kind='barh',rot=0, title="Incidents by day of the week", figsize=(15,10))