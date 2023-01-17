# Import pandas
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

df = pd.read_csv("../input/noaa-qclcd-2011/2011_Austin_Weather.txt")
df.head()
# Read the 2011_Austin_Weather.txt as a DataFrame attributing no header
df = pd.read_csv('../input/noaa-qclcd-2011/2011_Austin_Weather.txt', header=None)
df.head()
with open('../input/column-label/column_labels.txt') as file:
    column_labels = file.read()
    
# Split on the comma to create a list: column_labels_list    
    column_labels_list = column_labels.split(',')
                 
# Assign the new column labels to the DataFrame: df.columns
    df.columns = column_labels_list

df.head()
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
# Remove the appropriate columns: df_dropped
df_dropped = df.drop(list_to_drop, axis='columns')

# output of df_dropped.head()
print(df_dropped.head())
# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped.date.astype(str)
# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# Concatenate the new date and Time columns
date_string = df_dropped['date']+df_dropped['Time']

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)

# Print the output of df_clean.head()
df_clean.head()
df_clean.info()
# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-06-20 8:00':'2011-06-20 9:00', 'dry_bulb_faren'])
# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')

# Print the transformed dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-06-20 8:00':'2011-06-20 9:00', 'dry_bulb_faren'])

# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'], errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')
# Print the median of the dry_bulb_faren column
print(df_clean.dry_bulb_faren.median())

# Print the median of the dry_bulb_faren column for the time range '2011-Apr':'2011-Jun'
print(df_clean.loc['2011-Apr':'2011-Jun', 'dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the month of January
print(df_clean.loc['2011-Jan', 'dry_bulb_faren'].median())
# Read the file from input
df_climate_2010 = pd.read_csv('../input/weather_data_austin_2010/weather_data_austin_2010.csv')
#print(df_climate_2010)
# set.index() to df_climate_2010 for time series 
df_climate_2010.Date = pd.to_datetime(df_climate_2010.Date)
df_climate_2010.set_index(df_climate_2010.Date, inplace=True)
df_climate_2010_copy = df_climate_2010.copy()
df_climate_2010.head(2)
# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('D').mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate_2010 = df_climate_2010.resample('D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate_2010.reset_index()['Temperature']

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())
#print(df_clean)
# Select days that are sunny: sunny
sunny = df_clean.loc[df_clean['sky_condition'].str.contains('CLR')]

# Select days that are overcast: overcast
overcast = df_clean.loc[df_clean['sky_condition'].str.contains('OVC')]

# Resample sunny and overcast, aggregating by maximum daily temperature
sunny_daily_max = sunny.resample('D').max()
overcast_daily_max = overcast.resample('D').max()

# Print the difference between the mean of sunny_daily_max and overcast_daily_max
print(sunny_daily_max.mean() - overcast_daily_max.mean())
# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
visibility_temperature=df_clean[['visibility', 'dry_bulb_faren']]
weekly_mean = visibility_temperature.resample('W').mean()
# Print the output of weekly_mean.corr()
print(weekly_mean.corr())
df_clean.info()
df_clean['visibility'] = pd.to_numeric(df_clean['visibility'], errors='coerce')
visibility_temperature=df_clean[['visibility', 'dry_bulb_faren']]
weekly_mean = visibility_temperature.resample('W').mean()
# Print the output of weekly_mean.corr()
print(weekly_mean.corr())
# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots=True)
plt.show()
#a Boolean Series for sunny days: sunny
sunny = df_clean['sky_condition'] == 'CLR'
# Resample the Boolean Series by day and compute the sum: sunny_hours
sunny_hours = sunny.resample('D').sum()

# Resample the Boolean Series by day and compute the count: total_hours
total_hours = sunny.resample('D').count()

# Divide sunny_hours by total_hours: sunny_fraction
sunny_fraction = sunny_hours / total_hours

# Make a box plot of sunny_fraction
sunny_fraction.plot(kind='box')
plt.show()
# Resample dew_point_faren and dry_bulb_faren by Month, aggregating the maximum values: monthly_max
monthly_max = df_clean[['dew_point_faren', 'dry_bulb_faren']].resample('M').max()

# Generate a histogram with bins=8, alpha=0.5, subplots=True
monthly_max.plot(kind='hist', bins=8, alpha=0.5, subplots=True)

# Show the plot
plt.show()
# Extract the maximum temperature in August 2010 from df_climate: august_max
august_max = df_climate_2010_copy.loc['2010-08', 'Temperature'].max()
print('Max temperature registered in August 2010 was ' + str(august_max.max()))

# Resample the August 2011 temperatures in df_clean by day and aggregate the maximum value: august_2011
august_2011 = df_clean.loc['2011-Aug', 'dry_bulb_faren'].resample('D').max()
print('Max temperature registered in August 2011 was ' + str(august_2011.max()))

# Filter out days in august_2011 where the value exceeded august_max: august_2011_high
august_2011_high = august_2011.loc[august_2011 > august_max]


# Construct a CDF of august_2011_high
august_2011_high.plot(kind='hist', normed=True,cumulative=True, bins=25, linestyle='-', title='Probability of hotter day in August 2011')
plt.xlabel('Registered Temperature')

# Display the plot
plt.show()