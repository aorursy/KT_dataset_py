import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/partynyc/party_in_nyc.csv')
df.info()
df.shape
df.describe()
df.head(5)
df.drop(df[df['Borough'] == 'Unspecified'].index, inplace=True)

borough_stat = df['Borough'].value_counts()

borough_index = borough_stat.index

borough_values = borough_stat.values

plt.figure(figsize=(12,8))

borough_plot = sns.barplot(x=borough_values, y=borough_index, orient='h', palette='Greens_r')

plt.title('Where is the party?', size=22)

plt.xticks(rotation=45, ha='right', size=12)

plt.yticks(size=12)

plt.show()
location_stat = df['Location Type'].value_counts()

location_index = location_stat.index

location_values = location_stat.values

plt.figure(figsize=(12,8))

location_plot = sns.barplot(x=location_values, y=location_index, orient='h', palette='Blues_r')

plt.title('Location Type', size=22)

plt.xticks(rotation=45, ha='right', size=12)

plt.yticks(size=12)

plt.show()
plt.figure(figsize=(12,8))

bor_loc_plot = sns.countplot(df['Borough'], hue=df['Location Type'])

plt.title('Where is the party?', size=22)

plt.xticks(rotation=45, ha='right', size=12)

plt.yticks(size=12)

plt.show()
df = pd.read_csv('../input/partynyc/party_in_nyc.csv')
df['Created Date'] = pd.to_datetime(df['Created Date'])

df['Closed Date'] = pd.to_datetime(df['Closed Date'])

df['month'] = pd.DatetimeIndex(df['Created Date']).month #creates column 'year' stripping the year from datetime_zero

df['day'] = pd.DatetimeIndex(df['Created Date']).day

df['day'] = df['Created Date'].dt.day_name()

df['hour'] = pd.DatetimeIndex(df['Created Date']).hour

df
month_stat = df['month'].value_counts()

month_index = month_stat.index

month_values = month_stat.values

plt.figure(figsize=(12,8))

sns.set_style("darkgrid")

time_plot = sns.lineplot(x=month_index, y=month_values, palette='Greens_r')

plt.title('When is the party?', size=22)

plt.xticks(month_index, rotation=45, ha='right', size=12)

plt.yticks(size=12)

plt.xlabel('Month')

plt.ylabel('Number of Calls')

plt.show()
plt.figure(figsize=(12,8))

bor_loc_plot = sns.countplot(df['month'], hue=df['Borough'])

plt.title('Where and when is the party?', size=22)

plt.xticks( rotation=45, ha='right', size=12)

plt.yticks(size=12)

plt.xlabel('Month')

plt.ylabel('Number of Calls')

plt.show()
plt.figure(figsize=(12,8))

explode = (0.02, 0.02, 0, 0, 0, 0, 0)

labels= ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

df['day'].value_counts().plot(kind='pie', autopct='%.1f%%', textprops={'fontsize': 13}, explode=explode)

plt.title('Call Percentage by weekday', size=22)

plt.show()
day_stat = df['day'].value_counts()

day_index = day_stat.index

day_values = day_stat.values

plt.figure(figsize=(12,8))

day_plot = sns.barplot(x=day_values, y=day_index, orient='h', palette='Reds_r')

plt.title('Number of Calls per day of the Week', size=22)

plt.xticks(rotation=45, ha='right', size=12)

plt.yticks(size=12)

plt.xlabel('Number of Calls')

plt.ylabel('Day of the Week')

plt.show()
hour_stat = df['hour'].value_counts()

hour_index = hour_stat.index

hour_values = hour_stat.values

plt.figure(figsize=(12,8))

sns.set_style("darkgrid")

time_plot = sns.barplot(x=hour_index, y=hour_values, palette='hsv')

plt.title('When is the party?', size=22)

plt.xticks( rotation=45, size=12)

plt.yticks(size=12)

plt.show()
open_date = df['Created Date']

close_date = df['Closed Date']

df['duration'] = close_date - open_date

seconds  = df['duration'].apply(lambda x: x.total_seconds())

df['in_minutes'] = round(seconds/60, 1)

df
df= df[(df['in_minutes']>0)&(df['in_minutes']<24*60)]

df.dropna(inplace=True) # Getting rid of nan rows
plt.figure(figsize=(20,11))

df['in_minutes'].plot.hist(bins=20, figsize=(10,6), color='gold')

plt.title('Police response time in minutes', size=22)

plt.xticks(size=12)

plt.yticks(size=12)

plt.xlabel('Response Time', size=14)

plt.ylabel('Calls', size=14)

plt.show()
newdf = df.groupby('Borough')['in_minutes'].mean()

newdf
plt.figure(figsize=(12,8))

newdf.plot.barh(figsize=(10,6), color='darkturquoise')

plt.title('Average response time in minutes', size=22)

plt.xticks(size=12)

plt.yticks(size=12)

plt.xlabel('Average Time', size=14)

plt.ylabel('Borough', size=14)

plt.show()