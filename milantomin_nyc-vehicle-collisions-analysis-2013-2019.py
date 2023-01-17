# data processing

import pandas as pd
# Load NYPD Motor Vehicle Collisions file

df = pd.read_csv("/kaggle/input/nypd-motor-vehicle-collisions/nypd-motor-vehicle-collisions.csv")
df.info()
# Let's quickly analyze what we've got here ^

# 1.5 Million records

# 1 Million with Borough and Zip Code info >> in other words, we're missing 500k location records

# In case we have Location, we can do reverse Geocoding with GeoPY library and find out address (Borough + Zipcode)
# Column header titles cleaning/renaming



# Number of persons injured is the total of injured (pedestrians + cyclists + motorists)

# If the number is 0, it means 0 injures and 0 deaths in an incident, but it's still a record



df.rename(columns = {'ZIP CODE'          : 'ZIP_CODE',

                       'ON STREET NAME'    : 'STREET_ON',

                       'CROSS STREET NAME' : 'STREET_CROSS',

                       'OFF STREET NAME'   : 'STREET_OFF',

                       'NUMBER OF PERSONS INJURED'     : 'TOTAL_INJURED',

                       'NUMBER OF PERSONS KILLED'      : 'TOTAL_KILLED',

                       'NUMBER OF PEDESTRIANS INJURED' : 'PED_INJURED',

                       'NUMBER OF PEDESTRIANS KILLED'  : 'PED_KILLED',

                       'NUMBER OF CYCLIST INJURED'     : 'CYC_INJURED',

                       'NUMBER OF CYCLIST KILLED'      : 'CYC_KILLED',

                       'NUMBER OF MOTORIST INJURED'    : 'MOTO_INJURED',

                       'NUMBER OF MOTORIST KILLED'     : 'MOTO_KILLED',

                       'CONTRIBUTING FACTOR VEHICLE 1' : 'VEH_FACTOR_1',

                       'CONTRIBUTING FACTOR VEHICLE 2' : 'VEH_FACTOR_2',

                       'CONTRIBUTING FACTOR VEHICLE 3' : 'VEH_FACTOR_3',

                       'CONTRIBUTING FACTOR VEHICLE 4' : 'VEH_FACTOR_4',

                       'CONTRIBUTING FACTOR VEHICLE 5' : 'VEH_FACTOR_5',

                       'UNIQUE KEY' : 'UNIQUE_KEY',

                       'VEHICLE TYPE CODE 1' : 'VEH_TYPE_1',

                       'VEHICLE TYPE CODE 2' : 'VEH_TYPE_2',

                       'VEHICLE TYPE CODE 3' : 'VEH_TYPE_3',

                       'VEHICLE TYPE CODE 4' : 'VEH_TYPE_4',

                       'VEHICLE TYPE CODE 5' : 'VEH_TYPE_5'},

           inplace = True) 
# Missing values in columns

df.isna().sum()
# Borough and Zipcode are missing ~500k records >> ~30% which is significant and we can't disregard it

# I'll assign missing Borough records to NYC. It will be 5 boroughs and NYC to collect what's unassigned.



# Remove Total Injured and Total Killed NaN values

# TOTAL INJURED and TOTAL KILLED are > 0, otherwise it's justa a record, so let's keep only > 0 records
# Borough and Zipcode are missing ~500k records >> ~30% which is significant and we can't disregard it

# I'll assign missing Borough records to NYC. It will be 5 borougs and NYC to collect what's unassigned



# Remove Total Injured and Total Killed NaN values

# TOTAL INJURED and TOTAL KILLED are > 0, otherwise it's just a a record, so let's keep only > 0 records
# Fill all blank values in column Borough

# If a value is NaN it will be NYC

df.loc[df['BOROUGH'].isnull(), 'BOROUGH'] = 'NYC'
# Let's check it... BOROUGH should have 0 NaN values

df.isna().sum()
# Remove NaN from TOTAL INJURED

df = df.dropna(axis=0, subset=['TOTAL_INJURED'])
# Remove NaN from TOTAL KILLED

df = df.dropna(axis=0, subset=['TOTAL_KILLED'])
# Keep only > 0 values as df1

df1 = df[(df['TOTAL_INJURED'] > 0)]
# Keep only non-zero values as df2

df2 = df[(df['TOTAL_KILLED'] > 0)]
# Concatenate df1 and df2 and put it back as df; 0 values are now out

df = pd.concat([df1, df2])
# Combine DATE and TIME column to transform Series to DateTime needed for further analysis

df['DATE'] = df['DATE'] + ' ' + df['TIME']
# Convert string to DateTime

df['DATE'] = pd.to_datetime(df.DATE)
# Year filter

df['DATE_YEAR'] = pd.to_datetime(df['DATE']).dt.year
# Quarter filter

df['DATE_QUARTER'] = pd.to_datetime(df['DATE']).dt.quarter
# Month filter

df['DATE_MONTH'] = pd.to_datetime(df['DATE']).dt.month
# Day of the week filter

df['WEEKDAY'] = pd.to_datetime(df['DATE']).dt.weekday
df.info()
# We have 285,116 relevant records instead of 1.5 million and our file is 68 MB from 340 MB at the beginning

# This file is now even readable with Excel
import matplotlib.pyplot as plt

%matplotlib inline
# Year 2012 starts in July and for that reason it's incomplete and we can't use it in our analysis. 

# Let's filter out 2012 and leave 2019 just as a reference for a trend (today is mid-August 2019)

df = df[(df['DATE'] > '2013-01-01')]
plt.figure(figsize=(20, 25)).subplots_adjust(hspace = 0.4)



# Total number of PERSONS injured

plt.subplot(4, 2 ,1)

df.groupby('DATE_YEAR').TOTAL_INJURED.sum().plot.bar()

plt.title('Total number of PERSONS INJURED', fontsize=16)

plt.xlabel('Year', fontsize=13)



# Total number of MOTORISTS injured

plt.subplot(4, 2, 2)

df.groupby('DATE_YEAR').MOTO_INJURED.sum().plot.bar()

plt.title('Total number of MOTORISTS INJURED', fontsize=16)

plt.xlabel('Year', fontsize=13)



# Total number of CYCLISTS injury

plt.subplot(4, 2 ,3)

df.groupby('DATE_YEAR').CYC_INJURED.sum().plot.bar()

plt.title('Total number of CYCLISTS INJURED', fontsize=16)

plt.xlabel('Year', fontsize=13)



# Total number of PEDESTRIANS injured

plt.subplot(4, 2, 4)

df.groupby('DATE_YEAR').PED_INJURED.sum().plot.bar()

plt.title('Total number of PEDESTRIANS INJURED', fontsize=16)

plt.xlabel('Year', fontsize=13)



plt.show()
plt.figure(figsize=(20, 25)).subplots_adjust(hspace = 0.4)



# Total number of PERSONS killed

plt.subplot(4, 2 ,1)

df.groupby('DATE_YEAR').TOTAL_KILLED.sum().plot.bar()

plt.title('Total number of PERSONS KILLED', fontsize=16)

plt.xlabel('Year', fontsize=13)



# TTotal number of MOTORISTS killed

plt.subplot(4, 2, 2)

df.groupby('DATE_YEAR').MOTO_KILLED.sum().plot.bar()

plt.title('Total number of MOTORISTS KILLED', fontsize=16)

plt.xlabel('Year', fontsize=13)



# Total number of CYCLISTS killed

plt.subplot(4, 2 ,3)

df.groupby('DATE_YEAR').CYC_KILLED.sum().plot.bar()

plt.title('Total number of CYCLISTS KILLED', fontsize=16)

plt.xlabel('Year', fontsize=13)



# Total number of PEDESTRIANS killed

plt.subplot(4, 2, 4)

df.groupby('DATE_YEAR').PED_KILLED.sum().plot.bar()

plt.title('Total number of PEDESTRIANS KILLED', fontsize=16)

plt.xlabel('Year', fontsize=13)



plt.show()
fig, ax = plt.subplots(1, figsize=(25, 15))



plt.subplot(2, 2 ,1)

df.groupby('BOROUGH').TOTAL_INJURED.sum().sort_values(ascending=False).plot.bar()

plt.title('Number of people injured per borough', fontsize=18)

plt.xlabel('Borough,   *NYC = unknown location incidents', fontsize=14)



plt.subplot(2, 2 ,2)

df.groupby('BOROUGH').TOTAL_KILLED.sum().sort_values(ascending=False).plot.bar()

plt.title('Number of people killed per borough', fontsize=18)

plt.xlabel('Borough,   *NYC = unknown location incidents', fontsize=14)



plt.show()
# Total number of injured and killed per quarter

fig, ax = plt.subplots(1, figsize=(25, 15))



plt.subplot(2, 2 ,1)

df.groupby('DATE_QUARTER').TOTAL_INJURED.sum().plot.bar()

plt.title('Total number of PERSONS INJURED', fontsize=18)

plt.xlabel('Quarter', fontsize=14)



plt.subplot(2, 2 ,2)

df.groupby('DATE_QUARTER').TOTAL_KILLED.sum().plot.bar()

plt.title('Total number of PERSONS KILLED', fontsize=18)

plt.xlabel('Quarter', fontsize=14)



plt.show()
# Total number of injured and killed per quarter

fig, ax = plt.subplots(1, figsize=(25, 15))

plt.subplot(2, 2 ,1)

df.groupby('WEEKDAY').TOTAL_INJURED.sum().plot.bar()

plt.title('Total number of PERSONS INJURED per day of the week', fontsize=18)

plt.xlabel('Weekday,    0 = Sunday', fontsize=14)



plt.subplot(2, 2 ,2)

df.groupby('WEEKDAY').TOTAL_KILLED.sum().plot.bar()

plt.title('Total number of PERSONS KILLED per day of the week', fontsize=18)

plt.xlabel('Weekday,    0 = Sunday', fontsize=14)



plt.show()
#--------------------------------------------------------------------------------

# Example code that works:

# from geopy.geocoders import Nominatim

# geolocator = Nominatim(user_agent="geoapiExercises")

# from tqdm import tqdm

# tqdm.pandas()

# geolocator = Nominatim(user_agent="specify_your_app_name_here")

# from geopy.extra.rate_limiter import RateLimiter

# geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.0, max_retries=2, error_wait_seconds=5.0, swallow_exceptions=True, return_value_on_exception=None)

# df['ADDRESS'] = df['LOCATION'].progress_apply(geocode)



# The down side: it will return only ~1,000 addresses per day

# With GeoPY is possible to fill all NaN values in ZIP CODE and BOROUGH

# Example: 

# Input: 40.88939, -73.89839 

# Output: Broadway, Fieldston, The Bronx, Bronx County, NYC, New York, 10463, USA

#--------------------------------------------------------------------------------