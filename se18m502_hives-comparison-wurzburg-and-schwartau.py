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
temperature_data_w = pd.read_csv('../input/temperature_wurzburg.csv', sep = ',');

flow_data_w = pd.read_csv('../input/flow_wurzburg.csv', sep = ',');

humidity_data_w = pd.read_csv('../input/humidity_wurzburg.csv', sep = ',');

weight_data_w = pd.read_csv('../input/weight_wurzburg.csv', sep = ',');



temperature_data_s = pd.read_csv('../input/temperature_schwartau.csv', sep = ',');

flow_data_s = pd.read_csv('../input/flow_schwartau.csv', sep = ',');

humidity_data_s = pd.read_csv('../input/humidity_schwartau.csv', sep = ',');

weight_data_s = pd.read_csv('../input/weight_schwartau.csv', sep = ',');
temperature_time_arr_w = pd.to_datetime(temperature_data_w.timestamp)

ts_temperature_w = pd.Series(data=np.array(temperature_data_w.temperature), 

                           index=pd.DatetimeIndex(temperature_time_arr_w), dtype="float")



temperature_time_arr_s = pd.to_datetime(temperature_data_s.timestamp)

ts_temperature_s = pd.Series(data=np.array(temperature_data_s.temperature), 

                           index=pd.DatetimeIndex(temperature_time_arr_s), dtype="float")
ts_temperature_hour_w = ts_temperature_w.resample("H").mean()

ts_temperature_hour_s = ts_temperature_s.resample("H").mean()
plt.figure(1, figsize=(10,5), dpi=150)

plt.subplots_adjust(hspace = 0.9)

plt.subplot(211)

ts_temperature_hour_w.plot(title="Temperature per hour Wurzburg", color="red")



plt.subplot(212)

ts_temperature_hour_s.plot(title="Temperature per hour Schwartau", color="red")

plt.show()



flow_time_arr_w = pd.to_datetime(flow_data_w.timestamp)

ts_flow_w = pd.Series(data=np.array(flow_data_w.flow), 

                           index=pd.DatetimeIndex(flow_time_arr_w), dtype="float")



flow_time_arr_s = pd.to_datetime(flow_data_s.timestamp)

ts_flow_s= pd.Series(data=np.array(flow_data_s.flow), 

                           index=pd.DatetimeIndex(flow_time_arr_s), dtype="float")
#Same as before, resample hourly

ts_flow_hour_w = ts_flow_w.resample("H").sum()

ts_flow_hour_s = ts_flow_s.resample("H").sum()
ax = plt.figure(figsize=(10,4), dpi=200)

ts_flow_hour_w[ts_flow_hour_w < 0].plot(title="Flow per hour Wurzburg", color="green", label = "Departures")

ts_flow_hour_w[ts_flow_hour_w > 0].plot(color="orange", label = "Arrivals")

leg = ax.legend();



ax = plt.figure(figsize=(10,4), dpi=200)

ts_flow_hour_s[ts_flow_hour_s < 0].plot(title="Flow per hour Schwartau", color="green", label = "Departures")

ts_flow_hour_s[ts_flow_hour_s > 0].plot(color="orange", label = "Arrivals")

leg = ax.legend();

humidity_time_arr_w = pd.to_datetime(humidity_data_w.timestamp)

ts_humidity_w = pd.Series(data=np.array(humidity_data_w.humidity), 

                           index=pd.DatetimeIndex(humidity_time_arr_w), dtype="float")



humidity_time_arr_s = pd.to_datetime(humidity_data_s.timestamp)

ts_humidity_s = pd.Series(data=np.array(humidity_data_s.humidity), 

                           index=pd.DatetimeIndex(humidity_time_arr_s), dtype="float")
#Resample daile

ts_humidity_day_w = ts_humidity_w.resample("D").mean()

ts_humidity_day_s = ts_humidity_s.resample("D").mean()
plt.figure(1, figsize=(10,5), dpi=150)

plt.subplots_adjust(hspace = 0.9)

plt.subplot(211)

ts_humidity_day_w.plot(title="Humidity per day Wurzburg", color="blue")



plt.subplot(212)

ts_humidity_day_s.plot(title="Humidity per day Schwartau", color="blue")

plt.show()

weight_time_arr_w = pd.to_datetime(weight_data_w.timestamp)

ts_weight_w = pd.Series(data=np.array(weight_data_w.weight), 

                           index=pd.DatetimeIndex(weight_time_arr_w), dtype="float")



weight_time_arr_s = pd.to_datetime(weight_data_s.timestamp)

ts_weight_s = pd.Series(data=np.array(weight_data_s.weight), 

                           index=pd.DatetimeIndex(weight_time_arr_s), dtype="float")
#Resample daily taking the mean

ts_weight_day_w = ts_weight_w.resample("D").mean()

ts_weight_day_s = ts_weight_s.resample("D").mean()
plt.figure(1, figsize=(10,5), dpi=150)

plt.subplots_adjust(hspace = 0.9)

plt.subplot(211)

ts_weight_day_w.plot(title="Weight per day Wurzburg", color="black")



plt.subplot(212)

ts_weight_day_s.plot(title="Weight per day Schwartau", color="black")

plt.show()
