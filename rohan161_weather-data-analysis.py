import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import sys

from pandas import Series

import traceback

import time

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.



print(os.listdir("../input"))

data = pd.read_fwf('../input/weather-data/sample_weather.txt')

data.to_csv('weather.csv')

#data  = pd.read_csv('../input/Weather data/sample_weather.txt') #data= weather Information

data.info()
data.head(10)
print("describe: ")

print(data.describe())
fig = data.hist(bins=50, figsize=(20,15))
corr_matrix = data.corr()

attributes = ['690190','13910','51.75','22.0','28.9']

fig = scatter_matrix(data[attributes], figsize = (10,10), alpha = 1)
data.dropna(inplace=True)

data.drop_duplicates(inplace=True)

print("Success")
plt.figure(figsize=(20,10)) 

sns.heatmap(data.corr(),annot=True, cmap='cubehelix_r') 

plt.show()