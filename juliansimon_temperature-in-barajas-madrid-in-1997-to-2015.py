# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Data collected from https://www.wunderground.com/

# Location: Madrid, Barajas, LEMD

# Sample data daily from 1997 to 2015

dt_weather=pd.read_csv('../input/weather_madrid_LEMD_1997_2015.csv', delimiter=",")
# Convert to date

dt_weather.CET = pd.to_datetime(dt_weather.CET, format='%Y-%m-%d')

dt_weather['year'] = dt_weather.CET.dt.year

dt_weather['dayofyear'] = dt_weather.CET.dt.dayofyear
matplotlib.rcParams['figure.figsize'] = (12.0, 8.0)
dt_weather.head()
dt_weather.dtypes
# Heatmap with average temperatures by day of the year from 1997 to 2015

dt_temp = dt_weather.pivot("year", "dayofyear", 'Mean TemperatureC')

sns.heatmap(dt_temp, xticklabels=False)
# Mean temperature per year

dt_weather_year = dt_weather.groupby(['year'])['Mean TemperatureC'].mean()

dt_weather_year.plot(kind='bar')
# Plot data and a linear regression mean temperature per year

dt_weather_year = dt_weather.groupby(['year'])['year', 'Mean TemperatureC'].mean()

sns.regplot(x="year", y='Mean TemperatureC', data=dt_weather_year)
# Draw a plot of mean temperature and day of year

sns.jointplot(x="dayofyear", y="Mean TemperatureC", data=dt_weather, size=8)
dt_weather['Mean TemperatureC'].describe()
# Draw count points for Mean TemperatureC

sns.countplot(x="Mean TemperatureC", data=dt_weather)
# Counts plots minimum temperature

sns.countplot(x="Min TemperatureC", data=dt_weather)
# Maximum temperature

sns.countplot(x="Max TemperatureC", data=dt_weather)
print ("14.0", dt_weather[dt_weather['Mean TemperatureC'] == 14.0]['Mean TemperatureC'].count())

print ("15.0", dt_weather[dt_weather['Mean TemperatureC'] == 15.0]['Mean TemperatureC'].count())

print ("16.0", dt_weather[dt_weather['Mean TemperatureC'] == 16.0]['Mean TemperatureC'].count())