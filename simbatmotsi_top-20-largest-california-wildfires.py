# improrting libraries

import numpy as np # the mathematical manipulation library.

import pandas as pd # this will allow us to manipulate the csv



"""

locating the file in the kaggle kernel

"""

import os 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

library for visualization



The %matplotlib inline, is a magic function that lets the visual output

be displayed automatically within the notebook (inline).

'''

import matplotlib.pyplot as plt

%matplotlib inline
# loding the csv

data = pd.read_csv("/kaggle/input/top-20-largest-california-wildfires/top_20_CA_wildfires.csv",

                   header="infer", 

                   infer_datetime_format=True,

                   low_memory=True)
data.info()
data.head()
# looking into the unique causes if the fires

data['cause'].nunique()
data['cause'].unique()
data[['fire_name','deaths','year']]
plt.figure(figsize=(20,5))#x,y

plt.plot(data['fire_name'], data['deaths'])
plt.figure(figsize=(20,5))#x,y

plt.plot(data['fire_name'], data['deaths'], 'ro')
plt.barh(data['fire_name'], data['deaths'])
# class distribution of the fire causes

data.groupby('cause').size()
# class distribution

data.groupby('cause').size().plot(kind="barh")
data.groupby('year').size()
import pandas_profiling as pdp
report = pdp.ProfileReport(data, title = "Pandas Profiling Report")
report