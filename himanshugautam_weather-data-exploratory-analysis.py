#Importing required packages

from pathlib import Path

import pandas as pd

import numpy as np

import itertools

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans
weather = pd.read_csv('/kaggle/input/Beutenberg.csv', encoding='latin', parse_dates=['Date.Time'])

print("Rows:", weather.shape)

weather.head()
#Get Date & Hour from Date.Time Column

weather['Year'] = weather['Date.Time'].dt.year

weather['Month'] = weather['Date.Time'].dt.month

weather['Date'] = weather['Date.Time'].dt.date

weather['Hour'] = weather['Date.Time'].dt.hour

weather = weather[weather['Year'] <=2018]

print(weather.columns)

weather.head()
#Compute Hourly Avg Temp on monthly basis

w_temp = weather.pivot_table(index=['Year', 'Month'], columns=['Hour'], values=['T..degC.']) 

print(w_temp.shape)

w_temp.head()
#Lets draw a heatmap to visualize Seasons

import seaborn as sb

plt.figure(figsize=(20,10))

heat_map = sb.heatmap(w_temp, cmap=sb.color_palette("RdBu_r", 15), annot=True)
#Compute Hourly Sum Rains on monthly basis

w_rain = weather.pivot_table(index=['Year', 'Month'], columns=['Hour'], values=['rain..mm.'], aggfunc=np.sum)

plt.figure(figsize=(20,10))

heat_map = sb.heatmap(w_rain, cmap=sb.color_palette("Blues", 15), annot=True)

#TotalRain = list(w_rain.sum(axis=1))
#Compute Hourly avg Wind speed on monthly basis

w_wind = weather.pivot_table(index=['Year', 'Month'], columns=['Hour'], values=['wv..m.s.'], aggfunc=np.mean)

plt.figure(figsize=(20,10))

heat_map = sb.heatmap(w_wind, cmap=sb.color_palette("Blues", 15), annot=True)
#Compute Hourly avg CO2 on monthly basis

w_co2 = weather.pivot_table(index=['Year', 'Month'], columns=['Hour'], values=['CO2..ppm.'], aggfunc=np.mean)

plt.figure(figsize=(20,10))

heat_map = sb.heatmap(w_co2, cmap=sb.color_palette("Blues", 15))