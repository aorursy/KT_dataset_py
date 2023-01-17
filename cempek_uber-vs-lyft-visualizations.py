

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df_rides = pd.read_csv('../input/uber-lyft-cab-prices/cab_rides.csv')

df_weather = pd.read_csv('../input/uber-lyft-cab-prices/weather.csv')
# converting the timestamp data into real date format

df_rides['date'] = pd.to_datetime(df_rides['time_stamp']/ 1000, unit = 's')

df_weather['date'] = pd.to_datetime(df_weather['time_stamp'], unit = 's')
# Creating the new column that contain the location and 

df_rides['merged_date'] = df_rides['source'].astype('str') + ' - ' + df_rides['date'].dt.strftime('%Y-%m-%d').astype('str') + ' - ' + df_rides['date'].dt.hour.astype('str')

df_weather['merged_date'] = df_weather['location'].astype('str') + ' - ' + df_weather['date'].dt.strftime('%Y-%m-%d').astype('str') + ' - ' + df_weather['date'].dt.hour.astype('str')
#  df_rides['date'].dt.strftime('%m').head()

df_weather.index = df_weather['merged_date']
# Join the weather date on rides data

df_joined = df_rides.join(df_weather, on = ['merged_date'], rsuffix ='_w')
df_joined.info()
df_joined['id'].value_counts()
df_joined[df_joined['id'] == '29ce4d5f-07f9-4ce5-ada4-ae9ea49e92a6'].iloc[:,10:22]
id_group = pd.DataFrame(df_joined.groupby('id')['temp','clouds', 'pressure', 'rain', 'humidity', 'wind'].mean())

df_rides_weather = df_rides.join(id_group, on = ['id'])
# Creating the columns for Month, Hour and Weekdays 

df_rides_weather['Month'] = df_rides_weather['date'].dt.month

df_rides_weather['Hour'] = df_rides_weather['date'].dt.hour

df_rides_weather['Day'] =  df_rides_weather['date'].dt.strftime('%A')
# The distribution of rides in weekdays 

uber_day_count = df_rides_weather[df_rides_weather['cab_type'] == 'Uber']['Day'].value_counts()

uber_day_count = uber_day_count.reindex(index = ['Friday','Saturday','Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])

lyft_day_count = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft']['Day'].value_counts()

lyft_day_count = lyft_day_count.reindex(index = ['Friday','Saturday','Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])



fig , ax = plt.subplots(figsize = (12,12))

ax.plot(uber_day_count.index, uber_day_count, label = 'Uber')

ax.plot(lyft_day_count.index, lyft_day_count, label = 'Lyft')

ax.set(ylabel = 'Number of Rides', xlabel = 'Weekdays')

ax.legend()

plt.show()
# The ride distribution in one day 

fig , ax = plt.subplots(figsize= (12,12))

ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('Hour').Hour.count().index, df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('Hour').Hour.count(), label = 'Lyft')

ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('Hour').Hour.count().index, df_rides_weather[df_rides_weather['cab_type'] =='Uber'].groupby('Hour').Hour.count(), label = 'Uber')

ax.legend()

ax.set(xlabel = 'Hours', ylabel = 'Number of Rides')

plt.xticks(range(0,24,1))

plt.show()

# The Average price of rides by type of service



uber_order =[ 'UberPool', 'UberX', 'UberXL', 'Black','Black SUV','WAV' ]

lyft_order = ['Shared', 'Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Lux Black XL']

fig, ax = plt.subplots(2,2, figsize = (20,15))

ax1 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].name, y = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].price , ax = ax[0,0], order = uber_order)

ax2 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].name, y = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].price , ax = ax[0,1], order = lyft_order)

ax3 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('name').name.count().index, y = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('name').name.count(), ax = ax[1,0] ,order = uber_order)

ax4 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('name').name.count().index, y = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('name').name.count(), ax = ax[1,1],order = lyft_order)

for p in ax1.patches:

    ax1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

for p in ax2.patches:

    ax2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

ax1.set(xlabel = 'Type of Service', ylabel = 'Average Price')

ax2.set(xlabel = 'Type of Service', ylabel = 'Average Price')

ax3.set(xlabel = 'Type of Service', ylabel = 'Number of Rides')

ax4.set(xlabel = 'Type of Service', ylabel = 'Number of Rides')

ax1.set_title('The Uber Average Prices by Type of Service')

ax2.set_title('The Lyft Average Prices by Type of Service')

ax3.set_title('The Number of Uber Rides by Type of Service')

ax4.set_title('The Number of Lyft Rides by Type of Service')

plt.show()
# The average price by distance

fig , ax = plt.subplots(figsize = (12,12))

ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('distance').price.mean().index, df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('distance')['price'].mean(), label = 'Lyft')

ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('distance').price.mean().index, df_rides_weather[df_rides_weather['cab_type'] =='Uber'].groupby('distance').price.mean(), label = 'Uber')

ax.set_title('The Average Price by distance', fontsize= 15)

ax.set(xlabel = 'Distance', ylabel = 'Price' )

ax.legend()

plt.show()

# The average price by distance 

fig, ax = plt.subplots(1,2 , figsize = (20,5))

for i,col in enumerate(df_rides_weather[df_rides_weather['cab_type'] == 'Uber']['name'].unique()):

    ax[0].plot(df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean().index, df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean(), label = col)

ax[0].set_title('Uber Average Prices by Distance')

ax[0].set(xlabel = 'Distance in Mile', ylabel = 'Average price in USD')

ax[0].legend()

for i,col in enumerate(df_rides_weather[df_rides_weather['cab_type'] == 'Lyft']['name'].unique()):

    ax[1].plot(df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean().index, df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean(), label = col)

ax[1].set(xlabel = 'Distance in Mile', ylabel = 'Average price in USD')

ax[1].set_title('Lyft Average Prices by Distance')

ax[1].legend()

plt.show()

# the average rate per mile

df_rides_weather['rate_per_mile'] = round((df_rides_weather['price'] / df_rides_weather['distance'] ),2)

# The average rate per mile plot

fig, ax = plt.subplots(1,2,figsize = (12,5))

ax1 = sns.lineplot(x = df_rides_weather.groupby(['distance'])['rate_per_mile'].mean().index, y = df_rides_weather.groupby('distance')['rate_per_mile'].mean(), ax = ax[0])

ax2 = sns.lineplot(x = df_rides_weather.groupby(['distance'])['rate_per_mile'].mean().index, y = df_rides_weather.groupby('distance')['rate_per_mile'].mean(), ax = ax[1])

plt.xticks(range(0, 10,1))

ax1.set(xlabel = 'Distance', ylabel = 'Rate per Mile in USD')

ax2.set(xlabel = 'Distance', ylabel = 'Rate per Mile in USD', ylim = (0,15))

ax1.set_title('The Average Rate per Mile', fontsize = 16)

ax2.set_title('ZOOM Average Rate per Mile', fontsize = 16)

plt.show()
# Scatter chart for Rate per mile and distance

    # pivot table to calculate average rate based on cab_type, service type(name) and distance

rates_per_mile_pivot = df_rides_weather.pivot_table(index = ['cab_type', 'name', 'distance'] , values = ['rate_per_mile'])

rates_per_mile_pivot.reset_index(inplace = True)
fig, ax = plt.subplots(2,2, figsize = (20,8))

ax1 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['name'], ax = ax[0,0])

ax2 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['name'], ax = ax[1,0])

ax2.set( ylim = (0,20))

ax3 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['name'], ax = ax[0,1])

ax4 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['name'], ax = ax[1,1])

ax4.set( ylim = (0,20))

handles_uber, labels_uber = ax1.get_legend_handles_labels()

handles_uber = [handles_uber[6],handles_uber[3],handles_uber[4],handles_uber[5],handles_uber[1],handles_uber[2]]

labels_uber = [labels_uber[6],labels_uber[3],labels_uber[4],labels_uber[5],labels_uber[1],labels_uber[2]]

ax1.legend(handles_uber, labels_uber)

ax2.legend(handles_uber, labels_uber)

handles_lyft, labels_lyft = ax3.get_legend_handles_labels()

handles_lyft = [handles_lyft[6],handles_lyft[4],handles_lyft[5],handles_lyft[1],handles_lyft[2],handles_lyft[3]]

labels_lyft = [labels_lyft[6],labels_lyft[4],labels_lyft[5],labels_lyft[1],labels_lyft[2],labels_lyft[3]]

ax3.legend(handles_lyft, labels_lyft)

ax4.legend(handles_lyft, labels_lyft)

ax1.set_title('Uber Rate per Mile')

ax1.set(ylabel = 'Rate per Mile in USD', xlabel = ' ')

ax2.set_title('Uber Rate Zoom(0 to 20 USD)')

ax2.set(ylabel = 'Rate per Mile in USD', xlabel = 'Distance')

ax3.set_title('Lyft Rate per Mile')

ax3.set(ylabel = ' ', xlabel = ' ')

ax4.set_title('Lyft Rate Zoom(0 to 20 USD)')

ax4.set(ylabel = ' ', xlabel = 'Distance')

plt.show()
# Overrated rides

high_mile_rates = df_rides_weather[df_rides_weather['rate_per_mile'] > 80]

# The number of overrated rides by cab type

high_mile_rates['cab_type'].value_counts()
# Overrated Lyft rides

high_mile_rates[high_mile_rates['cab_type'] == 'Lyft'].loc[:,['distance', 'cab_type', 'price', 'surge_multiplier','name', 'rate_per_mile']]

# Overrated Uber Rides

high_mile_rates[high_mile_rates['cab_type'] == 'Uber'].loc[:,['distance', 'cab_type', 'price', 'surge_multiplier','name', 'rate_per_mile']].sort_values(by = 'rate_per_mile', ascending = False).head(20)
# The number of rides based on service type, distance, and price 

over_rated_pivot = high_mile_rates[high_mile_rates['cab_type'] == 'Uber'].pivot_table(index = ['name', 'distance', 'price'], values = ['id'], aggfunc = len).rename(columns = {'id' : 'count_rides'})

over_rated_pivot.reset_index(inplace =True)

over_rated_pivot.sort_values(by = ['count_rides', 'name'], ascending = False).head(15)