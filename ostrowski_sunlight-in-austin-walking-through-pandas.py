# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns; sns.set() # advanced visualization
import statsmodels.api as sm # advanced time series visualization
from pylab import rcParams # advanced time series visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Preliminary step reading 2011_Austin_Weather as DataFrame and inspecting df.head()
df = pd.read_csv('../input/noaa-2011-austin-weather/2011_Austin_Weather.txt')
df.head()
# Read the 2011_Austin_Weather.txt as a DataFrame attributing no header
df = pd.read_csv('../input/noaa-2011-austin-weather/2011_Austin_Weather.txt', header=None)
df.head()
# open() .txt file, .read() and .split() it 
# Attribute the splitted list to df.columns and inspect the new df.head()
with open('../input/column-label/column_labels.txt') as file:
    column_labels = file.read()
    column_labels_list = column_labels.split(',')
    df.columns = column_labels_list
df.head()
# Specify the list_to_drop with labels of columns to be dropped
# Drop the columns using df.drop() and inspecting the new df_dropped.head()
list_to_drop = ['sky_conditionFlag',
 'visibilityFlag',
 'wx_and_obst_to_vision',
 'wx_and_obst_to_visionFlag',
 'dry_bulb_farenFlag',
 'dry_bulb_celFlag',
 'wet_bulb_farenFlag',
 'wet_bulb_celFlag',
 'dew_point_farenFlag',
 'dew_point_celFlag',
 'relative_humidityFlag',
 'wind_speedFlag',
 'wind_directionFlag',
 'value_for_wind_character',
 'value_for_wind_characterFlag',
 'station_pressureFlag',
 'pressure_tendencyFlag',
 'pressure_tendency',
 'presschange',
 'presschangeFlag',
 'sea_level_pressureFlag',
 'hourly_precip',
 'hourly_precipFlag',
 'altimeter',
 'record_type',
 'altimeterFlag',
 'junk']
df_dropped = df.drop(list_to_drop, axis='columns')
df_dropped.head()
# Analyze columns left and data tipes using df_dropped.info()
df_dropped.info()
# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped['date'].astype(str)

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# Checking that both attributes date and Time changed from int64 to object
df_dropped[['date', 'Time']].info()
# Concatenate the new date and Time columns: date_string
# Checking the final format of date_string
date_string = df_dropped.date + df_dropped.Time
date_string.head(2)
# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)

# Print the output of df_clean.head()
df_clean.head()
# Getting .info() from our last df.clean data
df_clean.info()
# Print the relative_humidity temperature between 8 AM and 9 AM on June 20, 2011
df_clean.loc['Jun-20-2011 8:00' : 'Jun-20-2011 9:00', 'relative_humidity']
# Convert the dry_bulb_faren, wind_speed and dew_point_faren columns to numeric values
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'], errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')

# Drop date and Time columns since we alredy have them as index
# Inplace=True used to replace df_clean = df_clean.drop()
df_clean.drop(['date', 'Time'], axis='columns', inplace=True)
df_clean.head(2)
# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('D').mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011.dry_bulb_faren.values

# Check the array values on daily_temp_2011
print(daily_temp_2011[:5])

# Checking the length of daily_temp_2011 to see if it corresponds to a year
print('Legth of daily_temp_2011 array is', len(daily_temp_2011))
# Read the file from input
df_climate_2010 = pd.read_csv('../input/2010-austin-weather/weather_data_austin_2010.csv')

# Converting Date column .to_datetime()
# set.index() to df_climate_2010 for time series 
df_climate_2010.Date = pd.to_datetime(df_climate_2010.Date)
df_climate_2010.set_index(df_climate_2010.Date, inplace=True)

# We save df_climate_2010_raw copy for using later in Question 5
df_climate_2010_raw = df_climate_2010.copy()

df_climate_2010.info()
df_climate_2010.head(2)
# Downsample df_climate_2010 by day and aggregate by .mean(): daily_climate
df_climate_2010 = df_climate_2010.resample('D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate_2010
daily_temp_2010 = df_climate_2010.reset_index().Temperature.values

# Check the array values on daily_temp_2010
print(daily_temp_2010[:5])

# Checking the length of daily_temp_2010 to see if it corresponds to a year
print('Legth of daily_temp_2010 array is', len(daily_temp_2010))
# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_2010
print(difference.mean())
# Select days that are sunny: sunny
sunny = df_clean.loc[df_clean['sky_condition'].str.contains('CLR')]

# Select days that are overcast: overcast
overcast = df_clean.loc[df_clean['sky_condition'].str.contains('OVC')]

# Resample sunny and overcast, aggregating by maximum daily temperature
sunny_daily_max = sunny.resample('D').max()
overcast_daily_max = overcast.resample('D').max()

# Print the difference between the mean of sunny_daily_max and overcast_daily_max
print(sunny_daily_max.mean() - overcast_daily_max.mean())

#EXTRANOTE: For this slicing operation you want to use str.contains() and not 
#df_clean.loc[df['sky_condition'] == 'OVC'] because when inspecting the data OVC is a category
#with many variations such as OVC011, OVC 016 etc.
print('\nFurther explanation on why using .str.contains()\n',overcast.sky_condition[:5])
# Select the visibility and dry_bulb_faren columns: visibility_temperature
visibility_temperature = df_clean[['visibility', 'dry_bulb_faren']]

# .resample and checking .head()
weekly_mean = visibility_temperature.resample('w').mean()
weekly_mean.head(2)
visibility_temperature.info()
# Converting visibility from object pd.to_numeric() on the original set df_clean
# It is not a good practice to convert types on data slices or copies!
df_clean['visibility'] = pd.to_numeric(df_clean['visibility'], errors='coerce')
visibility_temperature = df_clean[['visibility', 'dry_bulb_faren']]

# .resample and checking .head()
weekly_mean = visibility_temperature.resample('w').mean()

# Print the output of weekly_mean.corr() using seaborn sns.heatmap()
plt.figure(figsize=(10,4))
ax = sns.heatmap(weekly_mean.corr(), vmin=-1, vmax=1, annot=True, linewidths=.5)

# Setting figure sizes for all figures from now on
# rcParams['figure.figsize'] could be used for individual size changes
# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots=True, figsize=(10,7))
plt.show()
# Checking attributes type()
print('Checking types of daily_temp attributes \n2010: %s\n2011: %s' % (type(daily_temp_2010), type(daily_temp_2011)))

# Redo the attributes with .resample()
# Extract the Temperature columns from both attributes
# Checking attributes type() and .shape after converting
df_climate_2010 = df_climate_2010.resample('D').mean()
daily_temp_2010 = df_climate_2010.Temperature
daily_mean_2011 = df_clean.resample('D').mean()
daily_temp_2011 = daily_mean_2011.dry_bulb_faren
print('\nChecking types of daily_temp attributes \n2010: %s\n2011: %s' % (type(daily_temp_2010), type(daily_temp_2011)))
print('\nChecking shapes of daily_temp attributes \n2010: %s\n2011: %s' % (daily_temp_2010.shape, daily_temp_2011.shape))

# Hotter days .plot()
plt.figure(figsize=(10,7))
daily_temp_2010.plot()
daily_temp_2011.plot()
plt.title('Non smoothed data')
plt.show()

# Smoothing data with a moving average using df.rolling().mean()
# Slicing smoothed_daily_temp_2011 in [50:] to remove mooving average effect
smoothed_daily_temp_2011 = daily_temp_2011.rolling(window=24).mean()[50:]
plt.figure(figsize=(10,7))
smoothed_daily_temp_2011.plot()
daily_temp_2010.plot()
plt.grid(True)
plt.title('Smoothed with Moving Average')
plt.show()
# Decomposition technique for further details on time series
decomposition = sm.tsa.seasonal_decompose(daily_temp_2011, model='additive')
rcParams['figure.figsize'] = 11, 9
fig = decomposition.plot()
plt.show()
# Retriving a Boolean Series for sunny days: sunny
sunny = df_clean['sky_condition'].str.contains('CLR')

# Resample the Boolean Series by day and compute the sum: sunny_hours
# Resample the Boolean Series by day and compute the count: total_hours
# Divide sunny_hours by total_hours: sunny_fraction
sunny_hours = sunny.resample('D').sum()
total_hours = sunny.resample('D').count()
sunny_fraction = sunny_hours / total_hours

# Make a box plot of sunny_fraction
fig, axes = plt.subplots(nrows=1, ncols=2)
plt.ylabel('Sun percentage')
fig1 = sunny_fraction.plot(kind='box', ax=axes[0], title='Fraction of sunny days');
fig1.set_ylabel('Fraction')
fig2 = sunny_hours.plot(kind='hist', ax=axes[1], title='Distribution of sunny days per sunny hours');
fig2.set_xlabel('Day hours')
rcParams['figure.figsize'] = 21, 12
plt.show()
# Extract the maximum temperature in August 2010 from df_climate_2010_raw: august_2010_max
# Note that we are using df_climate_2010_raw the df with hour based index
august_2010_max = df_climate_2010_raw.loc['Aug-2010', 'Temperature'].max()
print('Max temperature registered in August 2010 was ' + str(august_2010_max.max()))

# Resample the August 2011 temperatures in df_clean by day and aggregate the maximum value: august_2011_max
august_2011_max = df_clean.loc['Aug-2011', 'dry_bulb_faren'].resample('D').max()
print('Max temperature registered in August 2011 was ' + str(august_2011_max.max()))

# Filter out days in august_2011_max where the value exceeded august_2010_max: august_2011_high
august_2011_high = august_2011_max[august_2011_max > august_2010_max]

# Construct a CDF of august_2011_high
august_2011_high.plot(bins=25, kind='hist', normed=True, cumulative=True, figsize=(13,8), title='Cumulative Distribution Function - Probability of hotter day in August 2011')
plt.xlabel('Registered Temperature')
plt.axhline(y=0.5, color='r', linestyle='-')
plt.show()