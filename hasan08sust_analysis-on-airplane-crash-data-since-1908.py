from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

import os, sys
dataset = pd.read_csv("../input/airplane-crash-data-since-1908/Airplane_Crashes_and_Fatalities_Since_1908_20190820105639.csv")

dataset.shape
dataset.head()
dataset.columns = dataset.columns.str.lower()
dataset.head()
dataset.info()
operator = dataset['operator'].value_counts()

operator
locations = dataset['location'].value_counts()

locations
time = dataset['time'].value_counts()

print(time[time == 1].count())

print(time[time == 2].count())

print(time[time == 3].count())

print(time[time == 4].count())

print(time[time == 5].count())
time_count = time[time >4]



plt.figure(figsize=(10,8))

time_count.plot()

plt.ylabel("Frequency")

plt.xlabel("time count")

plt.grid(axis='y', alpha=0.75)

plt.show()
time15 = dataset[dataset['time'] == '15:00'][['time','location','operator','route', 'flight #']]

time15
time15['operator'].value_counts()
plt.figure(figsize=(12,8))

time15['operator'].value_counts().plot(kind='bar', title="Time15 Graph")

plt.show()
air_force = time15['operator'].value_counts().index.str.contains("Air Force",regex=True)

print("Air Force: {0}".format(time15['operator'].value_counts()[air_force].count()))

print("Private air company: {0}".format(time15['operator'].value_counts().count()-time15['operator'].value_counts()[air_force].count()))
time15.dropna(subset=['route'])
time15.groupby(['operator','location']).count()[['time','route']]
time17 = dataset[dataset['time'] == '17:00'][['time','location','operator','route']]

time17
time17['operator'].value_counts()

plt.figure(figsize=(12,8))

time17['operator'].value_counts().plot(kind='bar', title="Time17 Graph")

plt.show()
air_force = time17['operator'].value_counts().index.str.contains("Air Force",regex=True)

print("Air Force: {0}".format(time17['operator'].value_counts()[air_force].count()))

print("Private air company: {0}".format(time17['operator'].value_counts().count()-time17['operator'].value_counts()[air_force].count()))
accident_data = []

accident_data_bool = []



for time in time_count.index:

    time_data = dataset[dataset['time'] == time][['time','location','operator','route']]

    time_data_freq = time_data['operator'].value_counts()

    air_force = time_data_freq.index.str.contains("Air Force",regex=True)

    air_force_count = time_data['operator'].value_counts()[air_force].count()

    private_air_count = time_data['operator'].value_counts().count()-time_data['operator'].value_counts()[air_force].count()

    acc_dic = {"time": time, "Air Force":air_force_count,"private air company": private_air_count}

    accident_data.append(acc_dic)

    if air_force_count > private_air_count:

        accident_data_bool.append("Air Force")

    else:

        accident_data_bool.append("Private Air") 
accident_data[0:10]
accident_data_bool_ser = pd.Series(accident_data_bool)

accident_data_bool_ser.value_counts().plot.bar(figsize=(10,8), rot=45)

plt.ylabel("Occurrence Count")

plt.xlabel("Air Force & Private air company")

plt.title("Frequency Graph for Private air company and Air Force")

plt.show()