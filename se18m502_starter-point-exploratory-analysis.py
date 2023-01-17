import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns #Visualization

import matplotlib.pyplot as plt #Visualization

import random

import matplotlib.dates as mdates #dates

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

sns.set()



#print(os.listdir("../input"))
temperature_data = pd.read_csv('../input/temperature_2017.csv', sep = ',');

flow_data = pd.read_csv('../input/flow.csv', sep = ',');

humidity_data = pd.read_csv('../input/humidity.csv', sep = ',');

weight_data = pd.read_csv('../input/weight.csv', sep = ',');
temperature_data.info()

#print(temperature_data.head())

print(temperature_data.tail())
temperature_data.temperature.describe()
temperature_time_arr = pd.to_datetime(temperature_data.timestamp)

ts_temperature = pd.Series(data=np.array(temperature_data.temperature), 

                           index=pd.DatetimeIndex(temperature_time_arr), dtype="float")
ts_temperature_hour = ts_temperature.resample("H").mean()
#Is there any Nan?

ts_temperature_hour[ts_temperature_hour.isnull()]
#Manual correction for '2017-03-26 02:00:00', using the data of one hour before

ts_temperature_hour['2017-03-26 02:00:00'] = ts_temperature_hour['2017-03-26 01:00:00']

ts_temperature_hour[ts_temperature_hour.isnull()]
ax = plt.figure(figsize=(5,2), dpi=150).add_subplot(111)

ts_temperature_hour.plot(ax=ax, title="Temperature per hour", color="red")
#temperature_data.tail()
temperature_data_hourly = temperature_data[temperature_data['timestamp'].astype(str).str.match("[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1]) (2[0-3]|[01][0-9]):00:00")]

#temperature_data_hourly.tail()
#Tagging by sensor

tags_list = []

prev_timestamp = temperature_data_hourly.iloc[0].timestamp

tag_number = 1



for row in temperature_data_hourly.itertuples():

    current = row.timestamp

    if current < prev_timestamp:

        tag_number+=1

        

    tags_list.append(tag_number)

    prev_timestamp = current  



#print(len(tags_list))
#Add the sensor tag as a column

temperature_data_hourly['sensor'] = tags_list

#temperature_data_hourly
ax = plt.figure(figsize=(6,3), dpi=100).add_subplot(111)

temperature_data_hourly.pivot_table(index='timestamp',columns='sensor',values='temperature').plot(ax=ax, rot=45, linewidth=0.3)

ax.legend(loc='best',

          fancybox=True, ncol=2, fontsize=8)
temperature_data_hourly[temperature_data_hourly.temperature < -50].index;
temperature_data_hourly= temperature_data_hourly.drop([163499]);
#temperature_data_prueba.groupby('sensor').plot('timestamp', 'temperature')

ax = plt.figure(figsize=(6,3), dpi=100).add_subplot(111)

temperature_data_hourly.pivot_table(index='timestamp',columns='sensor',values='temperature').plot(ax=ax, rot=45, linewidth=0.3)

ax.legend(loc='lower center',

          fancybox=True, ncol=6, fontsize=8)
flow_data.info()

#flow_data.head()

flow_data.describe()
flow_time_arr = pd.to_datetime(flow_data.timestamp)

ts_flow = pd.Series(data=np.array(flow_data.flow), 

                           index=pd.DatetimeIndex(flow_time_arr), dtype="float")
#Same as before, resample hourly

ts_flow_hour = ts_flow.resample("H").sum()
#Is there any Nan?

ts_flow_hour[ts_flow_hour.isnull()]
ax = plt.figure(figsize=(6,3), dpi=200).add_subplot(111)

ts_flow_hour[ts_flow_hour < 0].plot(ax=ax, title="Flow per hour", color="green", label = "Departures")

ts_flow_hour[ts_flow_hour > 0].plot(ax=ax, color="orange", label = "Arrivals")

leg = ax.legend();
flow_departures = flow_data[flow_data.flow < 0]

flow_arrivals = flow_data[flow_data.flow > 0]

list_of_series = [flow_departures, flow_arrivals]



flow_departures['timestamp'] = pd.to_datetime(flow_departures['timestamp'])

flow_departures.index = flow_departures['timestamp']



flow_arrivals['timestamp'] = pd.to_datetime(flow_arrivals['timestamp'])

flow_arrivals.index = flow_arrivals['timestamp']



flow_arrivals = flow_arrivals.resample("M").sum()

flow_departures = flow_departures.resample("M").sum()

flow_departures['flow'] = flow_departures['flow'].abs()



#print(flow_arrivals)

#print(flow_departures)
width = 0.4

ax = plt.figure(figsize=(6,3), dpi=200).add_subplot(111)

flow_departures.plot(ax=ax, kind="bar",  color = "green", label = "departures", title = "In/Outs per month", width=width, position=1)

flow_arrivals.plot(ax=ax,kind="bar", color = "orange", label = "arrivals", width=width, position=0)

ax.legend(["departures", "arrivals"])

ax.set_xticklabels(['Jan 17', 'Feb 17', 'Mar 17', 'Apr 17', 'May 17', 'Jun 17', 'Jul 17', 'Aug 17', 'Sep 17', 'Oct 17', 'Nov 17', 'Dec 17',

                   'Jan 18', 'Feb 18', 'Mar 18', 'Apr 18', 'May 18', 'Jun 18', 'Jul 18', 'Aug 18', 'Sep 18', 'Oct 18', 'Nov 18', 'Dec 18',

                   'Jan 19', 'Feb 19', 'Mar 19', 'Apr 19', 'May 19'], rotation=90);
print(flow_arrivals[flow_arrivals.index == '2017-10-31'] > flow_departures[flow_departures.index == '2017-10-31'])

print(flow_arrivals[flow_arrivals.index == '2017-11-30'] > flow_departures[flow_departures.index == '2017-11-30'])

humidity_data.info()

humidity_data.head()

humidity_data.describe()
humidity_time_arr = pd.to_datetime(humidity_data.timestamp)

ts_humidity = pd.Series(data=np.array(humidity_data.humidity), 

                           index=pd.DatetimeIndex(humidity_time_arr), dtype="float")
#Resample hourly

ts_humidity_hour = ts_humidity.resample("H").mean()
#Is there any Nan?

ts_flow_hour[ts_flow_hour.isnull()]
ax = plt.figure(figsize=(5,2), dpi=200).add_subplot(111)

ts_humidity_hour.plot(ax=ax, title="Humidity per hour", color="blue")
ts_humidity_hour[ts_humidity_hour < 0]

ts_humidity_hour['2017-05-06 12:00:00'] = ts_humidity_hour['2017-05-06 11:00:00']

ts_humidity_hour['2017-05-06 13:00:00'] = ts_humidity_hour['2017-05-06 15:00:00']

ts_humidity_hour['2019-04-09 11:00:00'] = ts_humidity_hour['2019-04-09 10:00:00']

ts_humidity_hour['2019-04-09 12:00:00'] = ts_humidity_hour['2019-04-09 10:00:00']

ts_humidity_hour['2019-04-09 13:00:00'] = ts_humidity_hour['2019-04-09 15:00:00']

ts_humidity_hour['2019-04-09 14:00:00'] = ts_humidity_hour['2019-04-09 15:00:00']
ax = plt.figure(figsize=(5,2), dpi=200).add_subplot(111)

ts_humidity_hour.plot(ax=ax, title="Humidity per hour", color="blue")
weight_data.info()

weight_data.head()

weight_data.describe()
weight_time_arr = pd.to_datetime(weight_data.timestamp)

ts_weight = pd.Series(data=np.array(weight_data.weight), 

                           index=pd.DatetimeIndex(weight_time_arr), dtype="float")
#Resample hourly taking the mean

ts_weight_hour = ts_weight.resample("H").mean()
#Is there any Nan?

ts_weight_hour[ts_weight_hour.isnull()]
#Manual correction for '2017-03-26 02:00:00', using the data of one hour before

ts_weight_hour['2017-03-26 02:00:00'] = ts_weight_hour['2017-03-26 01:00:00']

ts_weight_hour[ts_flow_hour.isnull()]
ax = plt.figure(figsize=(5,2), dpi=200).add_subplot(111)

ts_weight_hour.plot(ax=ax, title="Weight per hour", color="black")
ts_weight_hour[ts_weight_hour < 30]
print(ts_weight_hour['2017-05-12 08:00:00'])

print(ts_weight_hour['2017-05-12 13:00:00'])



print(ts_weight_hour['2017-05-15 11:00:00'])

print(ts_weight_hour['2017-05-15 15:00:00'])



print(ts_weight_hour['2017-09-18 09:00:00'])

print(ts_weight_hour['2017-09-18 13:00:00'])

ts_weight_hour['2017-05-12 09:00:00'] = ts_weight_hour['2017-05-12 08:00:00']

ts_weight_hour['2017-05-12 10:00:00'] = ts_weight_hour['2017-05-12 08:00:00']

ts_weight_hour['2017-05-12 12:00:00'] = ts_weight_hour['2017-05-12 13:00:00']

ts_weight_hour['2017-05-12 11:00:00'] = ts_weight_hour['2017-05-12 13:00:00']

ts_weight_hour['2017-05-15 12:00:00'] = ts_weight_hour['2017-05-15 11:00:00']

ts_weight_hour['2017-05-15 13:00:00'] = ts_weight_hour['2017-05-15 15:00:00']

ts_weight_hour['2017-05-15 14:00:00'] = ts_weight_hour['2017-05-15 15:00:00']



ts_weight_hour['2017-09-18 10:00:00'] = ts_weight_hour['2017-09-18 09:00:00']

ts_weight_hour['2017-09-18 11:00:00'] = ts_weight_hour['2017-09-18 13:00:00']

ts_weight_hour['2017-09-18 12:00:00'] = ts_weight_hour['2017-09-18 13:00:00']
ax = plt.figure(figsize=(5,2), dpi=200).add_subplot(111)

ts_weight_hour.plot(ax=ax, title="Weight per hour", color="black")
#Resample monthly

ts_weight_monthly = ts_weight.resample("M").mean()

ts_humidity_monthly = ts_humidity.resample("M").mean()

ts_temperature_monthly = ts_temperature.resample("M").mean()

w_h_t = pd.DataFrame({'weight' : ts_weight_monthly, 'humidity': ts_humidity_monthly, 'temperature': ts_temperature_monthly})

w_h_t.plot(figsize=(10,6))
below_50 = ts_humidity_hour[ts_humidity_hour < 50]

brooding_humidity = ts_humidity_hour[(ts_humidity_hour > 50) & (ts_humidity_hour < 60)]

ax = plt.figure(figsize=(5,2), dpi=200).add_subplot(111)

ax.plot(ts_humidity_hour.index, ts_humidity_hour.values,color="blue", alpha=0.3)

#below_50.plot(ax=ax, color="yellow")

ax.scatter(below_50.index, below_50.values, color="red", s=1, alpha=1)

ax.scatter(brooding_humidity.index, brooding_humidity.values, color="yellow", s=1, alpha=1)

plt.xticks(rotation=90)

ax.tick_params(axis='both', which='major', labelsize=5)

ax.legend(["humidity", "low humidity", "brood rearing"],  prop={'size':5})

#brooding_humidity.index

brooding_temperature = ts_temperature_hour.filter(items=brooding_humidity.index)

brooding_temperature.head()
ax = plt.figure(figsize=(5,2), dpi=150).add_subplot(111)

ts_temperature_hour.plot(ax=ax, title="Temperature per hour", color="red", alpha= 0.2)

ax.scatter(brooding_temperature.index, brooding_temperature.values, color="yellow", s=1, alpha=1)

#According to literature temperature should be around 30ºC and 35ºC, let's take a +-5ºC tolerance

ax.axhline(y=25, color='blue', linestyle='--', linewidth=0.5)

ax.axhline(y=40, color='blue', linestyle='--', linewidth=0.5)

ax.legend(["temperature", "25ºC", "40ºC", "brooding temperature"],  prop={'size':5})

net_flow = ts_flow_hour.groupby(ts_flow_hour.index).sum()
net_flow_spring = net_flow[(net_flow.index > '2017-03-20 00:00:00') & (net_flow.index < '2017-06-21 23:59:59')]

#Lowest flow -> Indicates there was a significatly diference between arrivals and departures (more departures)

max_flow = net_flow_spring[net_flow_spring ==net_flow_spring.max()]

#Highest flow -> More arrivals

min_flow = net_flow_spring[net_flow_spring ==net_flow_spring.min()]

max_min_flow = pd.concat([min_flow, max_flow])

max_min_flow
#Resampling net flow weekly

weekly_flow = ts_flow_hour.resample("W-MON").sum()

weekly_flow.head()
#Find the biggest difference between two weeks

flow_difference = weekly_flow.diff()

positive_flow_weekly =  flow_difference[flow_difference == flow_difference.max()]

negative_flow_weekly =  flow_difference[flow_difference == flow_difference.min()]



print(positive_flow_weekly)

print(negative_flow_weekly)



#The maximum diference corresponds to more departures the week corresponding 

#to '2017-06-05' this week can correspond to the queen leaving with most part of the population
ax = plt.figure(figsize=(5,2), dpi=150).add_subplot(111)

ts_flow_hour.plot(ax=ax, title="Flow per hour", linewidth=0.5)

ax.axvspan('2017-06-05', '2017-06-12', color=sns.xkcd_rgb['grey'], alpha=0.5)

ax.axvline('2017-06-05', color='k', linestyle='--', linewidth=0.3)

ax.axvline('2017-06-12', color='k', linestyle='--', linewidth=0.3)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 

from mpl_toolkits.axes_grid1.inset_locator import mark_inset



ax = plt.figure(figsize=(5,2), dpi=150).add_subplot(111)

#ts_weight_hour.plot(ax=ax, title="Weight per week", linewidth=0.5)

ax.plot(ts_weight_hour.index, ts_weight_hour.values)

ax.axvspan('2017-06-05', '2017-06-12', color=sns.xkcd_rgb['grey'], alpha=0.5)

ax.axvline('2017-06-05', color='k', linestyle='--', linewidth=0.3)

ax.axvline('2017-06-12', color='k', linestyle='--', linewidth=0.3)



ax.tick_params(axis='both', which='major', labelsize=5)



#I want to select the x-range for the zoomed region. I have figured it out suitable values

# by trial and error. How can I pass more elegantly the dates as something like

x1 = pd.Timestamp('2017-06-05')

x2 = pd.Timestamp('2017-06-12')

y1 = 62

y2 = 70



axins = zoomed_inset_axes(ax, 4, loc=2) # zoom = 2

axins.plot(ts_weight_hour, color="orange", linewidth=0.4)

axins.set_xlim(x1, x2)

axins.set_ylim(y1, y2)

plt.yticks(visible=False)

plt.xticks(visible=False)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3")

plt.draw()

plt.show()