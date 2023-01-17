# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # used to visualise data

import seaborn as sns           # used to visualise data



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read in raw_csv and visulaise first 5 rows of data

raw_data = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv")

raw_data.head(5)
raw_data.dtypes
# First call raw_data and change to different variable name.

data_time = raw_data



# We will then convert our timestamp to a datetime value

data_time['timestamp'] = pd.to_datetime(data_time['timestamp'])



# Add two new columns for time of day and day of week. 

data_time['time'] = data_time['timestamp']

data_time['day'] = data_time['timestamp']



# View data to see columns were added

data_time.head(5)
# We can now check our dtypes again

data_time.dtypes
# Import datetime library for conversion of timestamp

import datetime



# Convert time value

data_time['time'] = data_time['time'].dt.hour



# Convert day value

data_time['day'] = data_time['day'].dt.weekday_name



data_time.head(5)
# Create groupby function for time of day

data_time_time = data_time.groupby('time').mean()



# Plot values calculated above

plt.figure()

plt.bar(data_time_time.index, data_time_time['cnt'])

plt.xlabel("Hour of Day")

plt.ylabel("Average Number of BikeShares")

plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22])

plt.suptitle("Bikeshares by Time of Day")

plt.show()
# Create groupby function for the day of the week

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

data_time_day = data_time.groupby('day').mean()

data_time_day = data_time_day.reindex(index = day_order)



# Plot values calculated above

plt.figure()

plt.bar(data_time_day.index, data_time_day['cnt'])

plt.xlabel("Day of Week")

plt.ylabel("Average Number of BikeShares")

plt.suptitle("Bikeshares by Day of the Week")

plt.xticks(rotation=90)

plt.ylim([900, 1300])

plt.show()
# Create a plot with 5 axes.

fig,(ax1, ax2, ax3, ax4, ax5)= plt.subplots(nrows=5)

fig.set_size_inches(18,25)



# Create all the subplots

sns.pointplot(data=data_time, x='time', y='cnt', ax=ax1)

sns.pointplot(data=data_time, x='time', y='cnt', hue='is_holiday', ax=ax2)

sns.pointplot(data=data_time, x='time', y='cnt', hue='is_weekend', ax=ax3)

sns.pointplot(data=data_time, x='time', y='cnt', hue='season', ax=ax4)

sns.pointplot(data=data_time, x='time', y='cnt', hue='weather_code',ax=ax5)
# Create a correlation matrix

corrmat = data_time.corr()

f, ax = plt.subplots(figsize = (10,10))

sns.heatmap(corrmat, vmax=1, annot=True);