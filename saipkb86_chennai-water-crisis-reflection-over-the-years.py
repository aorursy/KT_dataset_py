import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))

df_level = pd.read_csv('../input/chennai_reservoir_levels.csv')

df_rain = pd.read_csv('../input/chennai_reservoir_rainfall.csv')
df_level.head()
df_rain.head()
df_level.info()
df_rain.info()
# Convert date columns to date format & make them index

df_level['Date'] = pd.to_datetime(df_level['Date'],format='%d-%m-%Y')

df_rain['Date'] = pd.to_datetime(df_rain['Date'],format='%d-%m-%Y')

df_level.set_index('Date',drop=True,inplace=True)

df_rain.set_index('Date',drop=True,inplace=True)
# Check for any missing values

print(df_level.isna().sum())

print(df_rain.isna().sum())
# Observe the water levels over the years in each reservoir

import seaborn as sns

# Use seaborn style defaults and set the default figure size

sns.set(rc={'figure.figsize':(20, 4)})

plt.plot(df_level['POONDI'])

plt.plot(df_level['CHOLAVARAM'])

plt.plot(df_level['CHEMBARAMBAKKAM'])

plt.plot(df_level['REDHILLS'])

plt.legend(loc='best')

plt.title('Storage level in Reservoir')
sns.set(rc={'figure.figsize':(20, 7)})

plt.plot(df_rain['POONDI'])

plt.plot(df_rain['CHOLAVARAM'])

plt.plot(df_rain['CHEMBARAMBAKKAM'])

plt.plot(df_rain['REDHILLS'])

plt.legend(loc='best')

plt.title('Rainfall in Reservoir')
# Reservoir availability in percent of it's full capacity

sns.set(rc={'figure.figsize':(20, 4)})

plt.plot(df_level['POONDI']/3231)

plt.plot(df_level['CHOLAVARAM']/881)

plt.plot(df_level['CHEMBARAMBAKKAM']/3645)

plt.plot(df_level['REDHILLS']/3300)

plt.legend(loc='best')

plt.title('Reservoir Storage % in terms of full capacity')
# Create data frame for population

data = {'Year':[2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019],'Population':[7476986,7671388,7870844,8075486,8285448,8500870,8721893,8948662,9181327,9420041,9664963,9916252,10174074,10438600,10710004]}

df_population = pd.DataFrame(data)
df_population.head()
# Calculate change of water level in reservoir, per day. This can be considered as water used for drinking water needs 

# of the city

df_level_change = df_level.diff(periods=-1)
print(df_level.head())

print(df_level_change.head())
# Plot usage/water drawn from each reservoir 

# Reservoir availability in percent of it's full capacity

sns.set(rc={'figure.figsize':(20, 10)})

plt.plot(df_level_change['POONDI'])

plt.plot(df_level_change['CHOLAVARAM'])

plt.plot(df_level_change['CHEMBARAMBAKKAM'])

plt.plot(df_level_change['REDHILLS'])

plt.legend(loc='best')

plt.title('Water usage from each of the reservoir')
# Calculate total water drawn from all reservoirs per day & aggregate at Year level

df_level_change['Total_Water_Used'] = df_level_change['POONDI']+df_level_change['CHEMBARAMBAKKAM']+df_level_change['CHOLAVARAM']+df_level_change['REDHILLS']

df_level_change['Year'] = df_level_change.index

df_level_change['Quarter'] = df_level_change['Year'].dt.quarter

df_level_change['Year'] = df_level_change['Year'].dt.year
df_water_usage_year = pd.DataFrame(df_level_change.groupby('Year')[['Total_Water_Used','POONDI','CHEMBARAMBAKKAM','CHOLAVARAM','REDHILLS']].sum())

df_water_usage_quarter = pd.DataFrame(df_level_change.groupby(['Year','Quarter'])[['Total_Water_Used','POONDI','CHEMBARAMBAKKAM','CHOLAVARAM','REDHILLS']].sum())
df_water_usage_year.reset_index(inplace=True)

df_water_usage_quarter.reset_index(inplace=True)
# Merge with population data to check water usage 

df_water_usage_year = pd.merge(df_water_usage_year,df_population,on='Year')

df_water_usage_quarter = pd.merge(df_water_usage_quarter,df_population,on='Year')
plt.figure(figsize=(15,4))

plt.bar(df_population['Year'],df_population['Population'])

plt.title('Population Growth')

plt.xlabel('Year')

plt.ylabel('Population (10 million)')
# Plot stacked bar of water usage from each reservoir

df_water_usage_quarter_2 = pd.DataFrame(df_level_change.groupby(['Year','Quarter'])[['POONDI','CHEMBARAMBAKKAM','CHOLAVARAM','REDHILLS']].sum())

a = df_water_usage_quarter_2.plot(kind='bar',stacked=True)

plt.savefig('stacked.png')

plt.title('Water Usage from each Reservoir every Quarter')

plt.show()
# Calculate Water usage per person 

df_water_usage_year['Usage_per_person'] = df_water_usage_year['Total_Water_Used']/df_water_usage_year['Population']
plt.figure(figsize=(15,5))

plt.plot(df_water_usage_year['Year'],df_water_usage_year['Usage_per_person'], lw=4,c='red',ls='-.')

plt.xlabel('Year')

plt.ylabel('Usage per person (mcft)')

plt.title('Usage per person over years')