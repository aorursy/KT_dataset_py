# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading the dataset

env=pd.read_csv('/kaggle/input/environmental-variables-for-world-countries/World_countries_env_vars.csv')



#first look and the info about dataset

env.info()

env.head()
#correlation of dataset

env.corr()
#correlation with plot

f,ax=plt.subplots(figsize=(15,12))

sns.heatmap(env.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
#line plot

env.wind.plot(kind = 'line', label = 'wind', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.', figsize =(15,15))

env.temp_annual_range.plot(kind = 'line', color = 'r', label = 'temp_annual_range', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.', figsize =(15,15))

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')
#histogram of all columns

env.hist(figsize =(10,10))
#hexbin plot

#Actually, when I was searching the plot types, I came across this. But I do not understand this plot purpose.

env.plot(kind='hexbin', x='wind', y='temp_annual_range', color='g', figsize =(15,15))
env.head()
#minimum elevation

env.elevation.min()
#maximum elevation

env.elevation.max()
#minimum accessibility_to_cities

env.accessibility_to_cities.min()
#maximum accessibility_to_cities

env.accessibility_to_cities.max()
#the result of where elevation >400 and accessibility_to_cities < 100

env[(env['elevation']>400) & (env['accessibility_to_cities']<100)]
#printing the result of elevation where elevation >400 and accessibility_to_cities < 100 and cropland_cover<20

var1 = env[(env['elevation']>400) & (env['accessibility_to_cities']<100) & (env['cropland_cover']<20)]

for index, value in var1.iterrows():

    print (value.Country,"elevation: ", value.elevation)