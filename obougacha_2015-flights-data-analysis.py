import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

base_color = sns.color_palette()[0]
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
flights_df = pd.read_csv('/kaggle/input/flight-delays/flights.csv')

flights_df.head(2)
flights_df.shape
airports_df = pd.read_csv('/kaggle/input/flight-delays/airports.csv')

airports_df.head(2)
airports_df.shape
airlines_df = pd.read_csv('/kaggle/input/flight-delays/airlines.csv')

airlines_df.head(2)
airlines_df.shape
missing_values_df = pd.DataFrame()

missing_values_df['Feature'] = flights_df.columns

missing_values_df['N_missing'] = flights_df.isnull().sum().values

missing_values_df['M_percent'] = flights_df.isnull().sum().values*100/flights_df.shape[0]

missing_values_df
flights_df[flights_df['CANCELLED']==1].shape[0]
flights_df[flights_df['CANCELLED']==1].isnull().sum()
flights_df[flights_df['DIVERTED']==1].shape[0]
flights_df[flights_df['DIVERTED']==1].isnull().sum()
flights_df[flights_df['DEPARTURE_DELAY']>0].shape[0]
flights_df[flights_df['ARRIVAL_DELAY']>0].shape[0]
flights_df[flights_df['DEPARTURE_DELAY']>0].isnull().sum()
flights_df[flights_df['ARRIVAL_DELAY']>0].isnull().sum()
flights_df[flights_df['ARRIVAL_DELAY']>0].head(2)
flights_df.dtypes
flights_df.duplicated().sum()
canceled_flights = flights_df[flights_df['CANCELLED']==1]

diverted_flights = flights_df[flights_df['DIVERTED']==1]

canceled_flights.shape[0], diverted_flights.shape[0], flights_df.shape[0]
89884*100/5819079, 15187*100/5819079
cleaned_flights = flights_df.drop(canceled_flights.index)
cleaned_flights = cleaned_flights.drop(diverted_flights.index)
canceled_flights.reset_index(drop=True, inplace=True)

diverted_flights.reset_index(drop=True, inplace=True)

cleaned_flights.reset_index(drop=True, inplace=True)
canceled_flights.shape[0] + diverted_flights.shape[0]+ cleaned_flights.shape[0] == flights_df.shape[0]
canceled_flights.isnull().sum()*100/canceled_flights.shape[0]
cols = canceled_flights.isnull().sum()[canceled_flights.isnull().sum()>0].index.tolist()

cols.remove('SCHEDULED_TIME')
canceled_flights.drop(cols, axis=1,inplace=True)
canceled_flights.shape
canceled_flights['SCHEDULED_TIME'].mode()[0]
canceled_flights['SCHEDULED_TIME'] = canceled_flights['SCHEDULED_TIME'].fillna(85)
canceled_flights.isnull().sum()
diverted_flights.isnull().sum()*100/diverted_flights.shape[0]
cols=['ELAPSED_TIME', 'AIR_TIME', 'ARRIVAL_DELAY', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 

      'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
diverted_flights.drop(cols, axis=1, inplace=True)
diverted_flights.shape
diverted_flights[['SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'WHEELS_ON', 'TAXI_IN']]
diverted_flights['TAXI_IN'] = diverted_flights['TAXI_IN'].fillna(diverted_flights['TAXI_IN'].mode()[0])

diverted_flights['TAXI_IN'].isnull().sum()
arrival_delay = diverted_flights['ARRIVAL_TIME'] - diverted_flights['SCHEDULED_ARRIVAL']

arrival_delay.median()
diverted_flights['ARRIVAL_TIME']=diverted_flights.apply(lambda x: x['SCHEDULED_ARRIVAL']+237 if np.isnan(x['ARRIVAL_TIME']) else x['ARRIVAL_TIME'], axis=1)
diverted_flights['WHEELS_ON']=diverted_flights.apply(lambda x: x['ARRIVAL_TIME']-x['TAXI_IN'] if np.isnan(x['WHEELS_ON']) else x['WHEELS_ON'], axis=1)
diverted_flights.isnull().sum()
diverted_flights['SCHEDULED_TIME'].mode()[0]
diverted_flights['SCHEDULED_TIME'] = diverted_flights['SCHEDULED_TIME'].fillna(140)
diverted_flights.isnull().sum()
def fix_time(x): 

    if x%100>=60: 

        x=x+40

    if x//100>=24:

        x=x-2400

    return x
diverted_flights['ARRIVAL_TIME'] = diverted_flights['ARRIVAL_TIME'].apply(fix_time)
diverted_flights['WHEELS_ON'] = diverted_flights['WHEELS_ON'].apply(fix_time)
diverted_flights['WHEELS_ON'].describe()
cleaned_flights.isnull().sum()
cols= ['CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',

       'WEATHER_DELAY']

cleaned_flights = cleaned_flights.drop(cols, axis=1)
cleaned_flights.isnull().sum()
cleaned_flights = cleaned_flights.drop(['DIVERTED', 'CANCELLED'], axis=1)

diverted_flights = diverted_flights.drop(['DIVERTED', 'CANCELLED'], axis=1)

canceled_flights = canceled_flights.drop(['DIVERTED', 'CANCELLED'], axis=1)
('DIVERTED' in  cleaned_flights.columns, 'CANCELLED' in cleaned_flights.columns, 

 'DIVERTED' in  diverted_flights.columns, 'CANCELLED' in diverted_flights.columns, 

 'DIVERTED' in  canceled_flights.columns, 'CANCELLED' in canceled_flights.columns)
cleaned_flights.to_csv('canceled_flights.csv', index=False)

diverted_flights.to_csv('diverted_flights.csv', index=False)

canceled_flights.to_csv('canceled_flights.csv', index=False)
cleaned_flights['DEPARTURE_TIME'].hist(bins=1000)

plt.xlabel('Departure Time (HHMM)')

plt.ylabel('Count')

plt.show()
cleaned_flights['DEPARTURE_DELAY'].hist(bins=1000)

plt.xlabel('Departure Delay (Minutes)')

plt.ylabel('Count')

plt.show()
cleaned_flights['DEPARTURE_DELAY'].describe()
cleaned_flights['DEPARTURE_DELAY'].hist(bins=1000)

plt.xlabel('Departure Delay (Minutes)')

plt.xlim((-100,240))

plt.ylabel('Count')

plt.show()
cleaned_flights['SCHEDULED_DEPARTURE'].hist(bins=1000)

plt.xlabel('Scheduled Departure (HHMM)')

plt.ylabel('Count')

plt.show()
cleaned_flights['TAXI_OUT'].hist(bins=100)

plt.xlabel('The Duration Between Closing Gate and Wheels Out (Minutes)')

plt.ylabel('Count')

plt.show()
cleaned_flights['ELAPSED_TIME'].hist(bins=1000)

plt.xlabel('Duration between Gate Closing and Passenger Out (Minutes)')

plt.ylabel('Count')

plt.show()
cleaned_flights['AIR_TIME'].hist(bins=1000)

plt.xlabel('Flight Duration (Minutes)')

plt.ylabel('Count')

plt.show()
cleaned_flights['DISTANCE'].hist(bins=100)

plt.xlabel('Trip Distance (mi)')

plt.ylabel('Count')

plt.show()
cleaned_flights['ARRIVAL_TIME'].hist(bins=1000)

plt.xlabel('Arrival Time (HHMM)')

plt.ylabel('Count')

plt.show()
cleaned_flights['SCHEDULED_ARRIVAL'].hist(bins=1000)

plt.xlabel('Scheduled Arrival (HHMM)')

plt.ylabel('Count')

plt.show()
cleaned_flights['ARRIVAL_DELAY'].hist(bins=1000)

plt.xlabel('Arrival Delay (Minutes)')

plt.ylabel('Count')

plt.show()
cleaned_flights['TAXI_IN'].hist(bins=1000)

plt.xlabel('Landing Duration (Minutes)')

plt.ylabel('Count')

plt.show()
sns.catplot(x='MONTH', kind='count', data=cleaned_flights, color=base_color)

plt.show()
sns.catplot(x='DAY_OF_WEEK', kind='count', data=cleaned_flights, color=base_color)

plt.show()
cleaned_flights['ORIGIN_AIRPORT'].nunique(), cleaned_flights['DESTINATION_AIRPORT'].nunique()
origin_air_flights = cleaned_flights.groupby('ORIGIN_AIRPORT', as_index=False)['FLIGHT_NUMBER'].count()

origin_air_flights.sort_values(by='FLIGHT_NUMBER',inplace=True, ignore_index=True)
origin_air_flights.head(10)
origin_air_flights = origin_air_flights.merge(airports_df[['IATA_CODE', 'AIRPORT', 'STATE', 'COUNTRY']],

                                              right_on='IATA_CODE', left_on='ORIGIN_AIRPORT')
worst = origin_air_flights.iloc[:10,:]
sns.catplot(y='AIRPORT', x='FLIGHT_NUMBER', kind='bar', data=worst, 

            color=base_color, aspect=2)

plt.xlabel('Number of Flights')

plt.ylabel('Origin Airport Name')

plt.show()
best = origin_air_flights.iloc[-10:,:]
sns.catplot(y='AIRPORT', x='FLIGHT_NUMBER', kind='bar', data=best, 

            color=base_color, aspect=2)

plt.xlabel('Number of Flights')

plt.ylabel('Origin Airport Name')

plt.show()
dest_air_flights = cleaned_flights.groupby('DESTINATION_AIRPORT', as_index=False)['FLIGHT_NUMBER'].count()

dest_air_flights.sort_values(by='FLIGHT_NUMBER',inplace=True, ignore_index=True)
dest_air_flights = dest_air_flights.merge(airports_df[['IATA_CODE', 'AIRPORT', 'STATE', 'COUNTRY']],

                                              right_on='IATA_CODE', left_on='DESTINATION_AIRPORT')
worst = dest_air_flights.iloc[:10,:]
sns.catplot(y='AIRPORT', x='FLIGHT_NUMBER', kind='bar', data=worst, 

            color=base_color, aspect=2)

plt.xlabel('Number of Flights')

plt.ylabel('Destination Airport Name')

plt.show()
best = dest_air_flights.iloc[-10:,:]
sns.catplot(y='AIRPORT', x='FLIGHT_NUMBER', kind='bar', data=best, 

            color=base_color, aspect=2)

plt.xlabel('Number of Flights')

plt.ylabel('Destination Airport Name')

plt.show()
air_trips = cleaned_flights.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'], as_index=False)['FLIGHT_NUMBER'].count()

air_trips.sort_values(by='FLIGHT_NUMBER',inplace=True, ignore_index=True)
air_trips['Trips'] = air_trips.apply(lambda x: str(x['ORIGIN_AIRPORT'])+'-'+str(x['DESTINATION_AIRPORT']),axis=1)
sns.catplot(y='Trips', x='FLIGHT_NUMBER', kind='bar', 

            data=air_trips.iloc[-10:,:], 

            color=base_color, aspect=2)

plt.xlabel('Number of Flights')

plt.ylabel('Trips')

plt.show()
sns.scatterplot(x='SCHEDULED_DEPARTURE', y='DEPARTURE_DELAY', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Scheduled Departure (HHMM)')

plt.ylabel('Departure Delay in Minutes')

plt.show()
sns.scatterplot(x='DEPARTURE_TIME', y='DEPARTURE_DELAY', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Departure Time (HHMM)')

plt.ylabel('Departure Delay in Minutes')

plt.show()
sns.scatterplot(x='SCHEDULED_DEPARTURE', y='DEPARTURE_TIME', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Scheduled Departure (HHMM)')

plt.ylabel('Departure Time (HHMM)')

plt.show()
sns.scatterplot(x='SCHEDULED_DEPARTURE', y='TAXI_OUT', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Scheduled Departure (HHMM)')

plt.ylabel('Taxi Out in Minutes')

plt.show()
sns.scatterplot(x='TAXI_OUT', y='DEPARTURE_DELAY', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.ylabel('Departure Delay in Minutes')

plt.xlabel('Taxi Out in Minutes')

plt.show()
sns.scatterplot(x='SCHEDULED_DEPARTURE', y='SCHEDULED_TIME', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Scheduled Departure (HHMM)')

plt.ylabel('Estimated Flight Duration in Minutes')

plt.show()
sns.scatterplot(x='SCHEDULED_DEPARTURE', y='ELAPSED_TIME', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Scheduled Departure (HHMM)')

plt.ylabel('Elapsed Flight Duration in Minutes')

plt.show()
sns.scatterplot(x='SCHEDULED_DEPARTURE', y='AIR_TIME', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Scheduled Departure (HHMM)')

plt.ylabel('Flight Duration in Air in Minutes')

plt.show()
sns.scatterplot(x='DEPARTURE_TIME', y='AIR_TIME', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Departure Time (HHMM)')

plt.ylabel('Flight Duration in Air in Minutes')

plt.show()
sns.scatterplot(x='DEPARTURE_DELAY', y='AIR_TIME', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Departure Delay in Minutes')

plt.ylabel('Flight Duration in Air in Minutes')

plt.show()
sns.scatterplot(x='DISTANCE', y='AIR_TIME', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.show()
sns.scatterplot(y='TAXI_IN', x='SCHEDULED_ARRIVAL', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.ylabel('Duration of Landing in Minutes')

plt.xlabel('Scheduled Arrival (HHMM)')

plt.show()
sns.scatterplot(y='TAXI_IN', x='ARRIVAL_TIME', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.ylabel('Duration of Landing in Minutes')

plt.xlabel('Arrival Time (HHMM)')

plt.show()
sns.scatterplot(y='TAXI_IN', x='ARRIVAL_DELAY', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.ylabel('Duration of Landing in Minutes')

plt.xlabel('Arrival Delay in Minutes')

plt.show()
sns.scatterplot(y='DEPARTURE_DELAY', x='ARRIVAL_DELAY', data=cleaned_flights, alpha=0.2, linewidth=0)

plt.ylabel('Departure Delay in Minutes')

plt.xlabel('Arrival Delay in Minutes')

plt.show()
sns.catplot(x='DAY_OF_WEEK', y='DEPARTURE_DELAY', data=cleaned_flights, kind='violin', color=base_color)

plt.xlabel('Day of Week')

plt.ylabel('Departure Delay in Minutes')

plt.show()
sns.catplot(x='DAY_OF_WEEK', y='DEPARTURE_DELAY', data=cleaned_flights, kind='violin', color=base_color)

plt.xlabel('Day of Week')

plt.ylim((-50,200))

plt.ylabel('Departure Delay in Minutes')

plt.show()
cor = cleaned_flights[[x for x in cleaned_flights.columns if x not in ['YEAR', 'DAY', 'FLIGHT_NUMBER']]].corr()

cor
fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(1,1,1)

sns.heatmap(cor, ax=ax)

plt.show()
fig = plt.figure(figsize=(25,6))

ax1 = fig.add_subplot(1,5,1)

sns.scatterplot(x='TAXI_OUT', y='DEPARTURE_DELAY', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-1,0]],

                alpha=0.2, linewidth=0, ax=ax1)

plt.ylabel('Departure Delay in Minutes')

plt.xlabel('Taxi Out in Minutes')

plt.title(origin_air_flights.iloc[-1,3])

ax2 = fig.add_subplot(1,5,2)

sns.scatterplot(x='TAXI_OUT', y='DEPARTURE_DELAY', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-2,0]],

                alpha=0.2, linewidth=0, ax=ax2)

plt.ylabel('Departure Delay in Minutes')

plt.xlabel('Taxi Out in Minutes')

plt.title(origin_air_flights.iloc[-2,3])

ax3 = fig.add_subplot(1,5,3)

sns.scatterplot(x='TAXI_OUT', y='DEPARTURE_DELAY', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-3,0]],

                alpha=0.2, linewidth=0, ax=ax3)

plt.ylabel('Departure Delay in Minutes')

plt.xlabel('Taxi Out in Minutes')

plt.title(origin_air_flights.iloc[-3,3])

ax4 = fig.add_subplot(1,5,4)

sns.scatterplot(x='TAXI_OUT', y='DEPARTURE_DELAY', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-4,0]],

                alpha=0.2, linewidth=0, ax=ax4)

plt.ylabel('Departure Delay in Minutes')

plt.xlabel('Taxi Out in Minutes')

plt.title(origin_air_flights.iloc[-4,3])

ax5 = fig.add_subplot(1,5,5)

sns.scatterplot(x='TAXI_OUT', y='DEPARTURE_DELAY', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-5,0]],

                alpha=0.2, linewidth=0, ax=ax5)

plt.ylabel('Departure Delay in Minutes')

plt.xlabel('Taxi Out in Minutes')

plt.title(origin_air_flights.iloc[-5,3])

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(25,6))

ax1 = fig.add_subplot(1,5,1)

sns.scatterplot(x='TAXI_IN', y='ARRIVAL_DELAY', 

                data=cleaned_flights[cleaned_flights['DESTINATION_AIRPORT']==dest_air_flights.iloc[-1,0]],

                alpha=0.2, linewidth=0, ax=ax1)

plt.ylabel('Arrival Delay in Minutes')

plt.xlabel('Taxi In in Minutes')

plt.title(dest_air_flights.iloc[-1,3])

ax2 = fig.add_subplot(1,5,2)

sns.scatterplot(x='TAXI_IN', y='ARRIVAL_DELAY', 

                data=cleaned_flights[cleaned_flights['DESTINATION_AIRPORT']==dest_air_flights.iloc[-2,0]],

                alpha=0.2, linewidth=0, ax=ax2)

plt.ylabel('Arrival Delay in Minutes')

plt.xlabel('Taxi In in Minutes')

plt.title(dest_air_flights.iloc[-2,3])

ax3 = fig.add_subplot(1,5,3)

sns.scatterplot(x='TAXI_IN', y='ARRIVAL_DELAY', 

                data=cleaned_flights[cleaned_flights['DESTINATION_AIRPORT']==dest_air_flights.iloc[-3,0]],

                alpha=0.2, linewidth=0, ax=ax3)

plt.ylabel('Arrival Delay in Minutes')

plt.xlabel('Taxi In in Minutes')

plt.title(dest_air_flights.iloc[-3,3])

ax4 = fig.add_subplot(1,5,4)

sns.scatterplot(x='TAXI_IN', y='ARRIVAL_DELAY', 

                data=cleaned_flights[cleaned_flights['DESTINATION_AIRPORT']==dest_air_flights.iloc[-4,0]],

                alpha=0.2, linewidth=0, ax=ax4)

plt.ylabel('Arrival Delay in Minutes')

plt.xlabel('Taxi In in Minutes')

plt.title(dest_air_flights.iloc[-4,3])

ax5 = fig.add_subplot(1,5,5)

sns.scatterplot(x='TAXI_IN', y='ARRIVAL_DELAY', 

                data=cleaned_flights[cleaned_flights['DESTINATION_AIRPORT']==dest_air_flights.iloc[-5,0]],

                alpha=0.2, linewidth=0, ax=ax5)

plt.ylabel('Arrival Delay in Minutes')

plt.xlabel('Taxi In in Minutes')

plt.title(dest_air_flights.iloc[-5,3])

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(25,6))

ax1 = fig.add_subplot(1,5,1)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-1,0]],

                alpha=0.2, linewidth=0, ax=ax1)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-1,3])

ax2 = fig.add_subplot(1,5,2)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-2,0]],

                alpha=0.2, linewidth=0, ax=ax2)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-2,3])

ax3 = fig.add_subplot(1,5,3)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-3,0]],

                alpha=0.2, linewidth=0, ax=ax3)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-3,3])

ax4 = fig.add_subplot(1,5,4)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-4,0]],

                alpha=0.2, linewidth=0, ax=ax4)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-4,3])

ax5 = fig.add_subplot(1,5,5)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', 

                data=cleaned_flights[cleaned_flights['ORIGIN_AIRPORT']==origin_air_flights.iloc[-5,0]],

                alpha=0.2, linewidth=0, ax=ax5)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-5,3])

plt.tight_layout()

plt.show()
cleaned_flights['Trips'] = cleaned_flights.apply(lambda x: str(x['ORIGIN_AIRPORT'])+'-'+str(x['DESTINATION_AIRPORT']),axis=1)
t = air_trips[air_trips['ORIGIN_AIRPORT'].isin(origin_air_flights.iloc[-5:,0].values)].iloc[-20:,3].values.tolist()
aaa = cleaned_flights[cleaned_flights['Trips'].isin(t)]
fig = plt.figure(figsize=(25,6))

ax1 = fig.add_subplot(1,5,1)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', hue='Trips',

                data=aaa[aaa['ORIGIN_AIRPORT']==origin_air_flights.iloc[-1,0]],

                alpha=0.2, linewidth=0, ax=ax1, legend=False)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-1,3])

ax2 = fig.add_subplot(1,5,2)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', hue='Trips',

                data=aaa[aaa['ORIGIN_AIRPORT']==origin_air_flights.iloc[-2,0]],

                alpha=0.2, linewidth=0, ax=ax2, legend=False)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-2,3])

ax3 = fig.add_subplot(1,5,3)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', hue='Trips',

                data=aaa[aaa['ORIGIN_AIRPORT']==origin_air_flights.iloc[-3,0]],

                alpha=0.2, linewidth=0, ax=ax3, legend=False)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-3,3])

ax4 = fig.add_subplot(1,5,4)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', hue='Trips',

                data=aaa[aaa['ORIGIN_AIRPORT']==origin_air_flights.iloc[-4,0]],

                alpha=0.2, linewidth=0, ax=ax4, legend=False)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-4,3])

ax5 = fig.add_subplot(1,5,5)

sns.scatterplot(x='DISTANCE', y='AIR_TIME', hue='Trips',

                data=aaa[aaa['ORIGIN_AIRPORT']==origin_air_flights.iloc[-5,0]],

                alpha=0.2, linewidth=0, ax=ax5, legend=False)

plt.xlabel('Distance in Miles')

plt.ylabel('Flight Duration in Air in Minutes')

plt.title(origin_air_flights.iloc[-5,3])

plt.tight_layout()

plt.show()