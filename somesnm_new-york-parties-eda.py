%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import datetime

import matplotlib.pyplot as plt

import time
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',parse_dates=['pickup_datetime','dropoff_datetime'])

test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',parse_dates=['pickup_datetime'])

party = pd.read_csv('../input/partynyc/party_in_nyc.csv',parse_dates=['Created Date','Closed Date'])
# Time between recieving the call and police arrival

party['ticket_duration'] = (party['Closed Date'].view('int64') // 10**9 - party['Created Date'].view('int64') // 10**9)/60

# Dropping all the tickets with negative duration

party = party[(party['ticket_duration']>0)&(party['ticket_duration']<24*60)]

party.dropna(inplace=True) # Getting rid of nan rows
#Additional features

party['hour'] = party['Created Date'].dt.hour

train['hour'] = train['pickup_datetime'].dt.hour

party['dayofweek'] = party['Created Date'].dt.weekday_name

party['date'] = party['Created Date'].dt.date

party['epoch'] = party['Created Date'].view('int64') // 10**9 # Unix time

train['epoch'] = train['pickup_datetime'].view('int64') // 10**9

test['epoch'] = test['pickup_datetime'].view('int64') // 10**9
party.head()
party['Borough'].value_counts(ascending=True).plot(kind='barh',title='Nuber of calls by district',

                                                   figsize=(8,5));
party['ticket_duration'].plot.hist(title='Police response time in minutes',bins=20, figsize=(10,6));
party['ticket_duration'].hist(by=party['Borough'],figsize = (15,10),bins=20);
party['ticket_duration'].describe()
wds = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

party['dayofweek'].value_counts().reindex(wds).plot(kind='bar',title="Number of calls by weekday",figsize=(10,6));
party['hour'].value_counts(sort=False).plot(kind='bar',title="Number of calls by hour of the day",figsize=(10,6));
party['date'].value_counts().sort_index().plot(figsize=(12,6),

                                               title='Number of calls by day of the year');
city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85)

#fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True)

ax = plt.scatter(party['Longitude'].values, party['Latitude'].values,

              color='blue', s=0.5, label='train', alpha=0.1)

ax.axes.set_title('Coordinates of the calls')

ax.figure.set_size_inches(6,5)

plt.ylim(city_lat_border)

plt.xlim(city_long_border)

plt.show()
def match_party(row): 

    lat_range = 0.003 # 333 meters by latitude

    long_range = 0.004 ## 336 meters by longitude

    time_radius = 2*60*60 # 2 hours in seconds

    sl = party[party['epoch'].between(row['epoch']-time_radius,row['epoch']+time_radius)]

    sl = sl[sl['Longitude'].between(row['pickup_longitude']-long_range,row['pickup_longitude']+long_range)]

    sl = sl[sl['Latitude'].between(row['pickup_latitude']-lat_range,row['pickup_latitude']+lat_range)]

    return sl.shape[0]
train['num_complaints'] = pd.read_csv('../input/partynyc/train_parties.csv')['num_complaints']

test['num_complaints'] = pd.read_csv('../input/partynyc/test_parties.csv')['num_complaints']
train['num_complaints'].value_counts(normalize=True).head(10)
train[(train['hour']>18) | (train['hour']<6)]['num_complaints'].value_counts(normalize=True).head(10)
pd.read_csv('../input/partynyc/train_parties.csv').to_csv('train_parties.csv',index=False)

pd.read_csv('../input/partynyc/test_parties.csv').to_csv('test_parties.csv',index=False)