import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_file = ('../input/sunlight-in-austin-tx/2011_Austin_Weather.txt')
df = pd.read_csv(data_file)
df.head()
df.columns
# Read in the data file with header=None: df_headers
df_header = pd.read_csv(data_file, header = None)
df_header.head()
df.shape
column_labels = 'Wban,date,Time,StationType,sky_condition,sky_conditionFlag,visibility,visibilityFlag,wx_and_obst_to_vision,wx_and_obst_to_visionFlag,dry_bulb_faren,dry_bulb_farenFlag,dry_bulb_cel,dry_bulb_celFlag,wet_bulb_faren,wet_bulb_farenFlag,wet_bulb_cel,wet_bulb_celFlag,dew_point_faren,dew_point_farenFlag,dew_point_cel,dew_point_celFlag,relative_humidity,relative_humidityFlag,wind_speed,wind_speedFlag,wind_direction,wind_directionFlag,value_for_wind_character,value_for_wind_characterFlag,station_pressure,station_pressureFlag,pressure_tendency,pressure_tendencyFlag,presschange,presschangeFlag,sea_level_pressure,sea_level_pressureFlag,record_type,hourly_precip,hourly_precipFlag,altimeter,altimeterFlag,junk'
# Split on the comma to create a list: column_labels_list
column_labels_list = column_labels.split(',')
column_labels_list
# Assign the new column labels to the DataFrame: df.columns
df.columns = column_labels_list
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
df.head()
# Remove the appropriate columns: df_dropped
df_dropped= df.drop(list_to_drop, axis = 'columns')

# Print the output of df_dropped.head()
df_dropped.head()
type(df)
# Convert the date column to string: df_dropped['date']
df_dropped['date']= df_dropped['date'].astype(str)

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))
df_dropped.head()
# Concatenate the new date and Time columns: date_string
date_string = df_dropped['date']+df_dropped['Time']

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)
df_clean.head()
df_dropped.head()
# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
df_clean.loc['2011-06-20 08:00:00':'2011-06-20 09:00:00', 'dry_bulb_faren']
# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
df_clean['dry_bulb_faren']= pd.to_numeric(df_clean['dry_bulb_faren'], errors = 'coerce')

print(df_clean.loc['2011-06-20 08:00:00':'2011-06-20 09:00:00', 'dry_bulb_faren'])
# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'],errors = 'coerse')

df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'],errors = 'coerse')
# Print the median of the dry_bulb_faren column
print(df_clean['dry_bulb_faren'].mean())
# Print the median of the dry_bulb_faren column for the time range '2011-Apr':'2011-Jun'
print(df_clean.loc['2011-Apr':'2011-Jun' ,'dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the month of January
print(df_clean.loc['2011-Jan', 'dry_bulb_faren'].median())
df_climate = pd.read_csv('../input/weather_data_austin_2010/weather_data_austin_2010.csv',parse_dates = True , index_col = 'Date')
df_climate.head()
# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('D').mean()
daily_mean_2011.head() # mean value at 24 hours
# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values   #πάρε τις τιμές αυτής της στήλης

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate = df_climate.resample('D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate.reset_index()['Temperature']

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate 

print(difference.mean())
#Use .loc[] to select sunny days and assign to sunny. If 'sky_condition' equals 'CLR', then the day is sunny
# Select days that are sunny: sunny
#'ClR' = sunny day
sunny = df_clean.loc[df_clean['sky_condition'] == 'CLR']  

#Use .loc[] to select overcast days and assign to overcast. If 'sky_condition' contains 'OVC', then the day is overcast.
# Select days that are overcast: overcast
#'OVC' = outcast day
overcast = df_clean.loc[df_clean['sky_condition'].str.contains('OVC')] 

# Resample sunny and overcast, aggregating by maximum daily temperature
sunny_daily_max = sunny.resample('D').max()
overcast_daily_max = overcast.resample('D').max()

# Print the difference between the mean of sunny_daily_max and overcast_daily_max
print(sunny_daily_max.mean() - overcast_daily_max.mean())
df_clean['visibility'] = pd.to_numeric(df_clean['visibility'], errors = 'coerse') # ensure that the column 'visibilty'
                                                                                  # contains only numbers and NaN
# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
weekly_mean = df_clean[['visibility','dry_bulb_faren']].resample('W').mean()


# Print the output of weekly_mean.corr()
print(weekly_mean.corr())
# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots = True)
plt.show()
df_clean.columns
# looking for how many 'missing values' there are:
df_clean['visibility'].isnull().sum() 
df_clean['sky_condition'].head(20)  # 'ClR' = sunny day
# Create a Boolean Series for sunny days: sunny
sunny =df_clean['sky_condition'] == 'CLR'

# Resample the Boolean Series by day and compute the sum: sunny_hours
sunny_hours = sunny.resample('D').sum()
sunny_hours.head()
# Resample the Boolean Series by day and compute the count: total_hours
total_hours = sunny.resample('D').count()
total_hours.head()
# Divide sunny_hours by total_hours: sunny_fraction
sunny_fraction = sunny_hours / total_hours
sunny_fraction.head()
# Make a box plot of sunny_fraction
sunny_fraction.plot(kind='box')
plt.show()
# Resample dew_point_faren and dry_bulb_faren by Month, aggregating the maximum values: monthly_max
monthly_max = df_clean[['dew_point_faren','dry_bulb_faren']].resample('M').max()

# Generate a histogram with bins=8, alpha=0.5, subplots=True
monthly_max.plot(kind = 'hist', bins =8 , alpha = 0.5, subplots = True)
plt.show()
# Extract the maximum temperature in August 2010 from df_climate: august_max
august_max = df_climate.loc['2010-08','Temperature'].max()
print(august_max)
# Resample August 2011 temps in df_clean by day & aggregate the max value: august_2011
august_2011 = df_clean.loc['2011-Aug', 'dry_bulb_faren'].resample('D').max()
august_2011.head()
# Filter for days in august_2011 where the value exceeds august_max: august_2011_high
august_2011_high = august_2011.loc[august_2011>august_max]

# Construct a CDF of august_2011_high
august_2011_high.plot(kind = 'hist', density = True, cumulative = True, bins = 25 )
plt.show()

