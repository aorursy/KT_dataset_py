# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import csv files

power_wrt_region = pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/State_Region_corrected.csv', thousands = ',')



power_wrt_time = pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/file.csv', parse_dates = ['Date'], thousands = ',')
power_wrt_time.head()
power_wrt_time.info()
# Filling of NaN values

power_wrt_time.isnull().any()
# import missingo

import missingno as msn
msn.bar(power_wrt_time)
# Creates a matrix of missing values along with position

msn.matrix(power_wrt_time)
# Impute with 0

power_wrt_time.fillna(0, inplace = True)
# Northern Region Power

north_power = power_wrt_time[power_wrt_time.Region == 'Northern'].drop(['Region'], axis = 1)  

#north_power.set_index('Date', inplace = True)



# Southern Region Power

south_power = power_wrt_time[power_wrt_time.Region == 'Southern'].drop(['Region'], axis = 1) 

#south_power.set_index('Date', inplace = True)



# Eastern Region Power

east_power = power_wrt_time[power_wrt_time.Region == 'Eastern'].drop(['Region'], axis = 1) 

#east_power.set_index('Date', inplace = True)



# Western Region Power

west_power = power_wrt_time[power_wrt_time.Region == 'Western'].drop(['Region'], axis = 1) 

#west_power.set_index('Date', inplace = True)



# North Eastern Region Power

northeast_power = power_wrt_time[power_wrt_time.Region == 'NorthEastern'].drop(['Region'], axis = 1)

#northeast_power.set_index('Date', inplace = True)
north_power.head()
# Import necessary libraries

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()

%matplotlib inline
plt.figure(figsize = (20, 10))

sns.lineplot(x = 'Date', y = 'Thermal Generation Actual (in MU)', data = north_power)

north_power.plot(x = 'Date', figsize = (20, 10), title = 'All power Statistics for North Region')
south_power.plot(x = 'Date', figsize = (20, 10), title = 'All power Statistics for South Region')
plt.figure(figsize = (20 ,10))

sns.lineplot(x = power_wrt_time['Date'], 

             y =power_wrt_time['Nuclear Generation Actual (in MU)'],

             hue = 'Region', 

             markers = True,

             data = power_wrt_time,

             palette = sns.color_palette("mako_r", 5))
sns.barplot(x = power_wrt_time['Region'], 

             y =power_wrt_time['Nuclear Generation Actual (in MU)'],

             data = power_wrt_time)
east_power.plot(x = 'Date',y = 'Hydro Generation Actual (in MU)', figsize = (20, 10), kind='kde')
plt.figure(figsize = (20 ,10))

sns.boxplot(x = 'Region', y = 'Hydro Generation Actual (in MU)', data = power_wrt_time)
northeast_power.plot(x = 'Date', y = 'Thermal Generation Actual (in MU)',figsize = (20, 10), kind='kde')