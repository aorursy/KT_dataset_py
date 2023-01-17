import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
station = pd.read_csv("../input/station.csv", parse_dates=['installation_date'], index_col='id')
trip = pd.read_csv("../input/trip.csv", parse_dates=['start_date', 'end_date'], index_col='start_date')
weather = pd.read_csv("../input/weather.csv", parse_dates=['date'], index_col = 'date')
status = pd.read_csv("../input/status.csv", parse_dates = ['time'])
pd.merge(station, trip, left_index = True, right_on = 'start_station_id', how='inner').groupby('name')['name'].count().sort_values(ascending=False).head(5)
pd.merge(station, status[status.bikes_available==0], left_index=True, right_on='station_id', how = 'inner').groupby('name')['name'].count().sort_values(ascending=False).head(10)
counts = trip.groupby(['subscription_type'])['id'].count()

plt.pie(counts, explode=(0.1, 0), labels=['Customer', 'Subscriber'], colors=['tab:orange', 'tab:cyan'],
        autopct='%1.1f%%', 
        shadow = True)
plt.title('Customer vs Subscribers')
plt.axis('scaled')
plt.show()
pop_routes = trip.groupby(['start_station_id','end_station_id']).count()['id'].sort_values(ascending=False).head(5)
start_stations = pop_routes.index.get_level_values(0)
end_stations = pop_routes.index.get_level_values(1)
start_station_names = station.loc[start_stations]['name'].values
end_station_names = station.loc[end_stations]['name'].values
pd.DataFrame({'start_station': start_station_names, 'end_station': end_station_names, 'total_trips': pop_routes.values})
trip['start_hour'] = trip.index.hour
trip.plot(y='start_hour', kind='hist', range=[0, 23], bins=24, density=True, ax=plt.gca(), alpha=0.7, figsize=(12,6), title='usage by hour of day')
trip['day_of_week'] = trip.index.weekday
plt.figure(figsize=(12,6))
plt.xticks(np.arange(7), ('Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday', 'Sunday'))
trip.plot(y='day_of_week', kind='hist', range=[0, 6], bins=7, density=True, ax=plt.gca(), alpha=0.7, title='usage by day of week', rot = 90, rwidth=0.9)
trip.index.names
trip['weekend'] = trip.index.weekday//5
trip_duration = trip.groupby('weekend')['duration'].mean() // 60

fig = plt.figure(figsize=(6,6))
trip_duration.plot(kind='bar')
trip['duration_min'] = trip.duration//60
trip.groupby(['subscription_type']).mean()[['duration','duration_min']]
trip.groupby(['subscription_type']).median()[['duration','duration_min']]
trip.groupby(['start_station_name','subscription_type']).mean()['duration_min'].unstack().plot(kind='bar',
                                                                                               legend = True,figsize=(16,8),
                                                                                              title='Dration from type of subscription')
plt.ylabel('Duration min')
plt.show()