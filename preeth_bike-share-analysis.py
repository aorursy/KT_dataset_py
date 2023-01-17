import pandas as pd

%matplotlib inline

import random

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import calendar

from datetime import datetime
data = pd.read_csv("../input/bike.csv")

data.head()
data["date"] = data['Start Date'].apply(lambda x : x.split()[0])

data["hour"] = data['Start Date'].apply(lambda x : x.split()[1].split(":")[0]).astype(np.int32)

data["weekday"] = data.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%m/%d/%Y").weekday()])

data["month"] = data.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%m/%d/%Y").month])

data['month_int'] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%m/%d/%Y").month).astype(np.int32)

data['duration_min'] = (data['Duration']/60).astype(np.float32)
data.head()
station = pd.read_csv("../input/201408_station_data.csv")

station.head()
rides = pd.merge(data, station, how='left', left_on='Start Terminal', right_on='station_id')
rides.head()

rides = rides.drop(['station_id','name','installation'], axis=1)

rides=rides.rename(columns = {'lat':'start_lat'})

rides=rides.rename(columns = {'long':'start_long'})

rides=rides.rename(columns = {'dockcount':'start_dc'})

rides=rides.rename(columns = {'landmark':'start_landmark'})
rides.head()
rides = pd.merge(rides, station, how='left', left_on='End Terminal', right_on='station_id')

rides = rides.drop(['station_id','name','installation'], axis=1)

rides=rides.rename(columns = {'lat':'end_lat'})

rides=rides.rename(columns = {'long':'end_long'})

rides=rides.rename(columns = {'dockcount':'end_dc'})
rides=rides.rename(columns = {'landmark':'end_landmark'})
rides.head()
rides['start_landmark'].value_counts()/len(rides)
#print("Total no of days",len(set(rides['date']))

#print("Avg Minutes per Rides",sum(rides['duration_min'])/len(rides))

print("Total no of days %d Total no of rides %d Avg Minutes per Rides %f" %(len(set(rides['date'])),rides.shape[0],sum(rides['duration_min'])/len(rides)))
cus_type = sorted(rides['Subscriber Type'].value_counts()/len(rides))

print("Subscriber Type - Subscriber %f  and Customer %f " %(cus_type[0]*100,cus_type[1]*100))
d = pd.crosstab([rides['hour'],rides['duration_min'].sum()],rides['Subscriber Type'])[['Customer','Subscriber']]

d.index =d.index.droplevel(1)

d.plot(kind='bar',stacked=True,figsize=(8,8))
d = pd.crosstab([rides['start_landmark'],rides['Trip ID'].count()],rides['Subscriber Type'])[['Customer','Subscriber']]

d.index =d.index.droplevel(1)

d.plot(kind='pie',stacked=True,figsize=(12,6),subplots=True)
d = pd.crosstab([rides['start_landmark'],rides['Trip ID'].count()],rides['Subscriber Type'])[['Customer','Subscriber']]

d.index =d.index.droplevel(1)

d.plot(kind='bar',stacked=True,figsize=(12,6))
d = pd.crosstab([rides['weekday'],rides['Trip ID'].count()],rides['Subscriber Type'])[['Customer','Subscriber']]

d.index =d.index.droplevel(1)

d.plot(kind='bar',stacked=True,figsize=(8,8))
rides.groupby(['weekday'],sort=False)['duration_min'].agg(['sum']).plot(kind='bar')
data.groupby('weekday',sort=False)['Trip ID'].agg(['count']).plot(kind='bar')
trips = rides.groupby(['date'])['Trip ID'].count()

trips.plot(kind='line',figsize=(12,6))
d = pd.crosstab([rides['date'],rides['Trip ID'].count()],rides['Subscriber Type'])

d.index =d.index.droplevel(1)

d.plot(kind='line',stacked=True,figsize=(15,8))
d = pd.crosstab([rides['month'],rides['Trip ID'].count()],rides['Subscriber Type'])[['Customer','Subscriber']]

d.index =d.index.droplevel(1)

d.plot(kind='bar',stacked=True,figsize=(6,6),title='Number of Trips By Customer')
rides[rides['duration_min'] < 60.].groupby(['Subscriber Type'])['duration_min'].hist()
d = pd.crosstab([rides['Start Terminal'],rides['End Terminal']],rides['Trip ID'].count())

#d = d/ len(rides)

support = d.unstack()

support = support.T.reset_index(level=0, drop=True).T

support.head()

fig, ax = plt.subplots(figsize=(18,18))

sns.heatmap(support, mask=support.isnull())
d = pd.crosstab([rides['weekday'],rides['month']],rides['Trip ID'].count())

#d = d/ len(rides)

support = d.unstack()

support = support.T.reset_index(level=0, drop=True).T

support.head()

fig, ax = plt.subplots(figsize=(16,6))

sns.heatmap(support)
fig, ax = plt.subplots(figsize=(18,10))

sns.boxplot(rides['month'], rides['duration_min'], ax=ax, showfliers=False)

fig.autofmt_xdate()
fig, ax = plt.subplots(figsize=(18,10))

sns.boxplot(rides['weekday'], rides['duration_min'], ax=ax, showfliers=False)

fig.autofmt_xdate()
rides['weekday'].value_counts()/len(rides)
by_origin_state = rides.groupby('Start Terminal')

departure_dur_counts = by_origin_state['Trip ID'].count()

by_dest_state = rides.groupby('End Terminal')

arrival_dur_counts = by_dest_state['Trip ID'].count()

duration_df = pd.DataFrame([departure_dur_counts, arrival_dur_counts]).T

duration_df.plot(kind='bar', title='Number of duration by station',figsize=(15,6))
d = pd.crosstab([rides['date'],rides['Trip ID'].count()],rides['start_landmark'])

d.index =d.index.droplevel(1)

d.plot(kind='line',figsize=(15,8),subplots=True)
d = pd.crosstab([rides['date'],rides['Trip ID'].count()],rides['end_landmark'])

d.index =d.index.droplevel(1)

d.plot(kind='line',figsize=(15,8),subplots=True)
d = rides['Start Station'].value_counts()

d.plot(kind='bar',figsize=(15,5))
d = rides['End Station'].value_counts()

d.plot(kind='bar',figsize=(15,5))
d = rides[rides['start_landmark'] != rides['end_landmark']]

d = d.groupby(['start_landmark','end_landmark']).size().unstack().fillna(0)

sns.heatmap(d)

d = rides[rides['start_landmark'] != rides['end_landmark']]

d = pd.crosstab(d['weekday'],d['Subscriber Type'])

d.plot(kind='bar')
#unbalanced Stations

d = rides[rides['start_landmark'] == 'San Francisco']

d = d['Start Terminal'].value_counts()/len(d)

p = rides[rides['start_landmark'] == 'San Francisco']

p = p['End Terminal'].value_counts()/len(p)

f  = pd.concat([d, p], axis=1).fillna(0)

f.plot(kind='bar')
d = rides[rides['start_landmark'] == 'San Jose']

d = d['Start Terminal'].value_counts()

p = rides[rides['start_landmark'] == 'San Jose']

p = p['End Terminal'].value_counts()

f  = pd.concat([d, p], axis=1).fillna(0)

#f.plot(kind='bar')

f['result'] = (f['Start Terminal']-f['End Terminal'])/f['Start Terminal']

f['result'].plot(kind='bar')
d = rides[rides['start_landmark'] == 'San Francisco']

d = d['Start Terminal'].value_counts()

p = rides[rides['start_landmark'] == 'San Francisco']

p = p['End Terminal'].value_counts()

f  = pd.concat([d, p], axis=1).fillna(0)

#f.plot(kind='bar')

f['result'] = (f['Start Terminal']-f['End Terminal'])/f['Start Terminal']

#f = pd.merge(f, station, how='left', left_on='End Terminal', right_on='station_id')

#f['result'].plot(kind='bar')

f = f.reset_index()

f = pd.merge(f, station, how='left', left_on='index', right_on='station_id')

f = f[['result','name']]

f.index = f['name']

f.plot(kind='bar')
p = rides['Bike #'].value_counts()

p.hist()
d = rides[rides['start_landmark'] == rides['end_landmark']]

d = d.groupby(['start_landmark','end_landmark']).size().unstack().fillna(0)

sns.heatmap(d)
d = rides[rides['start_landmark'] == rides['end_landmark']]

d = pd.crosstab(d['weekday'],d['Subscriber Type'])

d.plot(kind='bar')
d = pd.crosstab([rides['start_landmark']],rides['Subscriber Type'])

d
d = rides['Start Station'].value_counts()

d[0:15]
d = rides['End Station'].value_counts()

d[0:15]
d = "From:"+rides['Start Station']+" To:"+rides['End Station']

d = d.value_counts()

d[0:15]