import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from bq_helper import BigQueryHelper

import warnings
warnings.filterwarnings('ignore')
# Input parameters
station_name = 'BANGKOK METROPOLIS'
years = range(2009, 2019)
helper = BigQueryHelper('bigquery-public-data', 'noaa_gsod')

sql = '''
SELECT
  year, mo, da, temp, min, max, prcp
FROM
    `bigquery-public-data.noaa_gsod.gsod{}` a
    INNER JOIN `bigquery-public-data.noaa_gsod.stations` b ON a.stn = b.usaf
WHERE
  country = 'TH' AND name = '{}'
'''

# Query weather data for each year
datasets = [ helper.query_to_pandas(sql.format(i, station_name)) for i in years ]

# print out each year's data shape
print('\n'.join([ '{}: {}'.format(x[0],x[1].shape) for x in zip(years, datasets)]))
# Concatenate datasets
weather = pd.concat(datasets)

# Handling missing values based on Table Schema description
weather['temp'] = weather['temp'].replace({ 9999.9 : np.nan })
weather['min'] = weather['min'].replace({ 9999.9 : np.nan })
weather['max'] = weather['max'].replace({ 9999.9 : np.nan })
weather['prcp'] = weather['prcp'].replace( { 99.99 : 0 })

weather.info()
# Data processing

# Setting date index
weather['date'] = weather.apply(lambda x: 
                                datetime.datetime(int(x.year), int(x.mo), int(x.da)), 
                                axis=1)
weather = weather.set_index('date')

# Convert temperature values from farenheit to celcius
def f_to_c(temp_f):
    temp_c = (temp_f - 32) * 5/9
    return round(temp_c, 2)

for col in ['temp','min','max']:
    weather[col] = weather[col].apply(f_to_c)

# Convert precipitation from inches to milimeters
def inch_to_mm(x):
    return round(x * 25.4)

weather['prcp'] = weather['prcp'].apply(inch_to_mm)
start_date = '{}0101'.format(years[0])
end_date = weather.index.max().strftime('%Y%m%d')

# Re-index to see if there are any days with missing data
weather = weather.reindex(pd.date_range(start_date, end_date))

# check if there is missing values occured from re-indexing
missing = weather[weather.isnull().any(axis=1)].index
# printing missing index
missing
# Interpolate numerical variables for the missing days
weather = weather.interpolate()

# Re-setting year, month, day fields
weather['year'] = weather.index.year
weather['mo'] = weather.index.month
weather['da'] = weather.index.day

# Verify interpolated data
weather.loc[missing].head(10)
weather[['temp','min','max','prcp']].describe()
data = weather[['temp','min','max','prcp']]
data.reset_index(inplace=True)
data.columns = ['Date','Avg Temp', 'Min Temp', 'Max Temp', 'Precip']
rank = data.sort_values('Min Temp').head(10).reset_index(drop=True)
rank.index = rank.index + 1
rank
rank = data.sort_values('Max Temp', ascending=False).head(10).reset_index(drop=True)
rank.index = rank.index + 1
rank
rank = data.sort_values('Precip', ascending=False).head(10).reset_index(drop=True)
rank.index = rank.index + 1
rank
plt.style.use('bmh')
weather[['max','temp','min','prcp']].plot(subplots=True, figsize=(18,12));
# melt to display
data = pd.melt(weather, 'year', ['temp','min','max'], 
               var_name='variable', value_name='degree')

plt.subplots(figsize=(15,8))

# Avg, Min, Max temp. boxplot
ax = sns.boxplot(x='year',y='degree',hue='variable', hue_order=['min','temp','max'],
           data=data)
ax.set_ylabel('Degree Celcius')
plt.show()

# Precip. boxplot
plt.subplots(figsize=(15,5))
ax  = sns.boxplot(x='year', y='prcp', data=weather)
ax.set_ylabel('Precipitation (mm)');
ax.set_yscale('log')
year_df = weather.groupby('year').mean()

f, axes = plt.subplots(nrows=2, figsize=(10,10))
ax = year_df[['temp','min','max']].plot(ax=axes[0]);
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax.set_ylabel('Temperature (C)')
ax.set_title('Avarage Daily Temperature by Year');

ax = year_df['prcp'].plot(ax=axes[1]);
ax.set_ylabel('Precipitation (mm)')
ax.set_title('Average Daily Precipitation by Year');

plt.tight_layout();
# for labeling
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

data = weather.groupby(['mo'], as_index=False)[['temp','min','max','prcp']].mean()

f, axes = plt.subplots(nrows=2, figsize=(10,10))
ax = data[['temp','min','max']].plot(ax=axes[0]);
ax.set_ylabel('Temperature (C)')
ax.set_xlabel('Month')
ax.set_xticks(np.arange(0,12))
ax.set_xticklabels(months)
ax.set_title('Average Daily Temperature by Month');

ax = data['prcp'].plot(ax=axes[1]);
ax.set_ylabel('Precipitation (mm)')
ax.set_xlabel('Month')
ax.set_xticks(np.arange(0,12))
ax.set_xticklabels(months)
ax.set_title('Average Daily Precipitation by Month');

plt.tight_layout();
month_df = weather.groupby(['year','mo'], as_index=False)[['temp','prcp']].mean()

# Temperature heatmap
data = month_df.pivot('year','mo','temp')
data.columns = months

plt.subplots(figsize=(10,5))
sns.heatmap(data, cmap='YlOrRd',annot=True, fmt='.1f', vmin=27)
plt.title('Average Daily Temperature (C) by Month')
plt.yticks(rotation=0)
plt.show()

# Precipitation heatmap
data = month_df.pivot('year','mo','prcp')
data.columns = months

plt.subplots(figsize=(10,5))
sns.heatmap(data, cmap='Blues',annot=True, fmt='.1f')
plt.title('Average Daily Precipitation (mm) by Month')
plt.yticks(rotation=0);
# slicing only for 2017, 2018 and until November as we don't have data for December
condition = np.logical_and(weather['year'].isin([2017,2018]), weather['mo'] <= 11)
data = weather[condition]

plt.subplots(figsize=(12,6))

sns.boxplot(x='mo', y='temp', hue='year', data=data)
plt.xlabel('Month')
plt.ylabel('Temerature (C)')
plt.title('Temperature Comparison');
plt.subplots(figsize=(12,6))
sns.barplot(x='mo',y='prcp', hue='year', data=data, ci=None)
plt.xlabel('Month')
plt.ylabel('Precipitation (mm)')
plt.title('Precipitation Comparison');
month_begins = ['0101','0201','0301','0401','0501','0601','0701','0801','0901','1001','1101','1201']
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

data = weather[weather['year'].isin([2017,2018])]
data['monthday'] = data.index.strftime('%m%d')
data = data.pivot('monthday', 'year','temp').dropna()

window = 7
data = data.rolling(window, min_periods=1).mean()

d2017 = data.loc[:, 2017]
d2018 = data.loc[:, 2018]

f, ax = plt.subplots(figsize=(20,5))

# Plot 2017
plt.plot(data.index, d2017, label='2017', color='silver')
plt.xticks(month_begins[:11] ,months[:11])

# Plot 2018
plt.plot(data.index, d2018, label='2018', color='k')
plt.xticks(month_begins[:11], months[:11])

# Where 2018 is higher than 2017 - filling a warm color
plt.fill_between(data.index, d2017, d2018, where= d2018 >= d2017,
                facecolor='coral') 

# Where 2018 is lower than 2017 -  filling a cool color
plt.fill_between(data.index, d2017, d2018, where= d2018 < d2017,
                facecolor='steelblue')

plt.title('{}-Day Moving Average Comparison (2017 vs 2018)'.format(window))
plt.ylabel('Temperature (C)')
plt.xlabel('Date')

import matplotlib.patches as mpatches
handles, labels = ax.get_legend_handles_labels()
handles = handles + [mpatches.Patch(color='coral'), mpatches.Patch(color='steelblue')]
labels = labels + ['2018 > 2017','2018 < 2017']
plt.legend(handles=handles, labels=labels);
