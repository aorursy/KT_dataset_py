# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Separating the files that can't be opened because of unknown errors

valid_filenames = list()
invalid_filenames = list()

for filename in os.listdir("../input"):
    try:
        pd.read_csv('../input/'+ filename)
        valid_filenames.append('../input/'+ filename)
    except:
        invalid_filenames.append('../input/'+ filename)
        
print('Valid filenames')
print(pd.Series(valid_filenames))

print('\nInvalid filenames')
print(pd.Series(invalid_filenames))
# # A peek into Data
# for filename in valid_filenames:
#     print("\n",filename)
#     print(pd.read_csv('../input/'+ filename).head(2))
# Getting Two combined datasets of Uber Data: Apr-Sept,2014 and Jan-June,2015 
df_april = pd.read_csv('../input/uber-raw-data-apr14.csv')
df_may = pd.read_csv('../input/uber-raw-data-may14.csv')
df_june = pd.read_csv('../input/uber-raw-data-jun14.csv')
df_july = pd.read_csv('../input/uber-raw-data-jul14.csv')
df_aug = pd.read_csv('../input/uber-raw-data-aug14.csv')
df_sept = pd.read_csv('../input/uber-raw-data-sep14.csv')

# Concatenating Files Apr-Sept, 2014
df_aprsept_2014 = pd.DataFrame()
df_aprsept_2014 = df_aprsept_2014.append(df_april)
df_aprsept_2014 = df_aprsept_2014.append(df_may)
df_aprsept_2014 = df_aprsept_2014.append(df_june)
df_aprsept_2014 = df_aprsept_2014.append(df_july)
df_aprsept_2014 = df_aprsept_2014.append(df_aug)
df_aprsept_2014 = df_aprsept_2014.append(df_sept)
df_aprsept_2014.head(10)
# Changing Time values to Pandas Time
df_aprsept_2014['Date/Time'] = pd.to_datetime(df_aprsept_2014['Date/Time'],format = '%m/%d/%Y %H:%M:%S')
# Function for Expanding DateTime to separate columns
def expand_date(df,date_column, inplace = True):
    if inplace:
        df['Date/Time'] = pd.to_datetime(df[date_column])
    else:
        df['Date/Time'] = pd.to_datetime(df[date_column].copy())        
    df['date'] = df['Date/Time'].dt.date
    df['month'] = df['Date/Time'].dt.month
    df['week'] = df['Date/Time'].dt.week
    df['MonthDayNum'] = df['Date/Time'].dt.day
    df['HourOfDay'] = df['Date/Time'].dt.hour
    df['DayOfWeekNum'] = df['Date/Time'].dt.dayofweek
    df['DayOfWeek'] = df['Date/Time'].dt.weekday_name
    return df
# Expanding Date and Time using the Function that I created
expand_date(df_aprsept_2014,'Date/Time',inplace=True);
df_aprsept_2014.groupby('MonthDayNum').count()['Base'].plot(kind='bar', figsize = (8,6))
plt.ylabel('Total Journeys')
plt.title('Journeys by Month Day');
df_aprsept_2014.groupby('DayOfWeekNum').count()['Base'].plot(kind='bar', figsize = (8,6))
plt.ylabel('Total Journeys')
plt.title('Journeys by Month Day');
df_aprsept_2014.groupby('HourOfDay').count()['Base'].plot(kind='bar', figsize = (8,6))
plt.ylabel('Total Journeys')
plt.title('Journeys by Month Day');
# Finding out monthwise distribution with dispatching base number
df_aprsept_2014.groupby(['Base','month']).count().unstack('Base')['Date/Time'].plot(kind='bar', figsize = (8,6),stacked=True)
plt.ylabel('Total Journeys')
plt.title('Journeys by Month');
# Finding the change in popularity of each Dispatching Base by month name
df_aprsept_2014.groupby(['Base','month']).count().unstack('Base')['Date/Time'].plot(figsize = (8,6))
plt.ylabel('Total Journeys')
plt.title('Journeys by Month by Dispatch Base');
# We can see that B02682 grew more popular in the months to come whereas B02616 lost some popularity
df2 = pd.read_csv('../input/Uber-Jan-Feb-FOIL.csv')

# Extracting Date and Month values for effective groupby plots
df2.date = pd.to_datetime(df2.date)
df2['DayOfWeekNum'] = df2['date'].dt.dayofweek
df2['DayOfWeek'] = df2['date'].dt.weekday_name
df2['MonthDayNum'] = df2['date'].dt.day
df2['month'] = df2['date'].dt.month
df2.columns
# Date vs Active Vehicles
df2.groupby('DayOfWeek')['active_vehicles'].sum().plot(kind='bar', figsize = (8,6))
# Finding the ratio of trips/active_vehicles
df2['trips/vehicle'] = df2['trips']/df2['active_vehicles']
df2.head(5)
# Date-wise Demand vs Supply Chart
df2.set_index('date').groupby(['dispatching_base_number'])['trips/vehicle'].plot(legend = True, figsize = (11,8),grid = True);
plt.ylabel('Average trips/vehicle')
plt.title('Demand vs Supply chart (Date-wise)');
# Day-wise Demand vs Supply Chart
df2.groupby(['DayOfWeek'])['trips/vehicle'].mean().plot(kind = 'bar')
plt.ylabel('Average trips/vehicle')
plt.title('Demand vs Supply chart (Day-wise)');

# On an average, Saturdays had the most trips per vehicle.
# Which suggests that there was a greater chance of finding a ride on a Saturday than other days
# Month-wise Demand vs Supply Chart
df2.groupby(['month'])['trips/vehicle'].mean().plot(kind = 'bar')
plt.ylabel('Average trips/vehicle')
plt.title('Demand vs Supply chart (Month-wise)');

# On an average, February observed greater rides per vehicle than January
# Demand vs Supply Chart of January
df2[df2.month == 1].groupby(['MonthDayNum'])['trips/vehicle'].mean().plot(kind = 'bar')
plt.ylabel('Average trips/vehicle')
plt.title('Demand vs Supply chart (January)');
# Demand vs Supply Chart of February
df2[df2.month == 2].groupby(['MonthDayNum'])['trips/vehicle'].mean().plot(kind = 'bar')
plt.ylabel('Average trips/vehicle')
plt.title('Demand vs Supply chart (February)');
# Average Daily Rides between the months
df2.groupby(['MonthDayNum','month']).sum().unstack('month')['trips'].plot(kind='bar', figsize = (13,9))
plt.ylabel('Average trips')
plt.title('Average Daily Rides in January and February');
df2.shape[0]
df_aprsept_2014.head(10)
df_aprsept_2014.groupby('month')['Date/Time'].count().plot(kind = 'bar');
plt.title('Ride Density by Month');
plt.ylabel("Number of Rides");
df_aprsept_2014['weekday'] = False
print("Working with weekday")
df_aprsept_2014.loc[df_aprsept_2014.DayOfWeekNum>=5,'weekday'] = False
df_aprsept_2014.loc[df_aprsept_2014.DayOfWeekNum<5,'weekday'] = True
df_aprsept_2014.groupby('weekday')['Date/Time'].count().plot(kind = 'bar', figsize = (3,5));
plt.title('Ride Density by Weekend/Weekday');
plt.ylabel("Number of Rides");
weekday_names = ['0:Monday, 1:Tuesday, 2:Wednesday, 3:Thursday, 4:Friday, 5:Saturday, 6:Sunday']
print(weekday_names)
df_aprsept_2014.groupby('DayOfWeekNum')['Date/Time'].count().plot(kind = 'bar', figsize = (3,5));
plt.title('Ride Density by Individual Days in Week');
plt.ylabel("Number of Rides");