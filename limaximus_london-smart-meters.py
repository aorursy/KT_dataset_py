# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_list = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if '.csv' in filename and '.gz' not in filename:
            file_list.append(os.path.join(dirname, filename))

print(len(file_list))

hhblock_files = [i for i in file_list if '/hhblock_dataset/b' in i]
halfhour_files = [j for j in file_list if '/halfhourly_dataset/b' in j]
daily_files = [d for d in file_list if '/daily_dataset/b' in d]
print(len(hhblock_files))
print(len(halfhour_files))
print(len(daily_files))
# daily dataset
dd = pd.read_csv('/kaggle/input/smart-meters-in-london/daily_dataset.csv/daily_dataset.csv')
print(dd.shape)
print(dd.head())
# hhblock: records here are unique on LCLid + day
# other columns are meter readings every half hour
#hh = pd.concat((pd.read_csv(f) for f in hhblock_files))
#print(hh.shape)

# another view of half-hour readings
# rows are unique on LCLid and timestamp
#halfhour = pd.concat((pd.read_csv(g) for g in halfhour_files))
#print(halfhour.shape)
# same as dd file, just split into blocks
daily = (pd.concat(pd.read_csv(d) for d in daily_files))
print(daily.shape)
daily['day'] = pd.to_datetime(daily['day'])
# read household info
ih = pd.read_csv('/kaggle/input/smart-meters-in-london/informations_households.csv')
print(ih.shape)
print(ih.Acorn.unique())
print(ih.Acorn_grouped.unique())
ih.head()
daily = pd.merge(daily, ih, on="LCLid", how="left")
print(daily.head())
#plots

#sns.distplot(daily['energy_sum'], kde = False)

histograms = sns.FacetGrid(daily, col='Acorn_grouped', col_wrap=3)
histograms = histograms.map(sns.distplot, 'energy_sum', bins=30)
plt.show()
histograms = sns.FacetGrid(daily, col='Acorn_grouped', col_wrap=3)
histograms = histograms.map(sns.distplot, 'energy_mean', bins=30)
plt.show()
histograms = sns.FacetGrid(daily, col='stdorToU', col_wrap=2)
histograms = histograms.map(sns.distplot, 'energy_sum', bins=30)
plt.show()
daily.groupby('stdorToU').agg(group=('stdorToU', 'first'),
                              avg_daily_energy=('energy_sum', 'mean'),
                              med_daily_energy=('energy_sum', 'median')
                             )
daily.groupby(['Acorn_grouped', 'stdorToU']).agg(group=('stdorToU', 'first'),
                              avg_daily_energy=('energy_sum', 'mean'),
                              med_daily_energy=('energy_sum', 'median')
                             )
print(daily.groupby('Acorn_grouped').size().describe())
print(daily.groupby('file').size().describe())
print(daily.groupby('LCLid').size().describe())

housedays = daily[['LCLid', 'day']].drop_duplicates()
print(housedays.groupby('LCLid').size().describe())
hhb4 = pd.read_csv('/kaggle/input/smart-meters-in-london/hhblock_dataset/hhblock_dataset/block_4.csv')

print(hhb4.describe())
hhb4.head()
hh_b4 = pd.read_csv('/kaggle/input/smart-meters-in-london/halfhourly_dataset/halfhourly_dataset/block_4.csv')
print(hh_b4.dtypes)
hh_b4.head(30)
daily_weather = pd.read_csv('/kaggle/input/smart-meters-in-london/weather_daily_darksky.csv')
daily_weather['date'] = pd.to_datetime(daily_weather['temperatureMaxTime'])
print(daily_weather.shape)
print(daily_weather.dtypes)
daily_weather.head()
print(daily_weather.date.min())
hweather = pd.read_csv('/kaggle/input/smart-meters-in-london/weather_hourly_darksky.csv')
hweather['date'] = pd.to_datetime(hweather['time'])
print(hweather.dtypes)
print(hweather.shape)
print(hweather.date.min())
print(hweather.date.max())
hweather.head()
acorn_deets = pd.read_csv('/kaggle/input/smart-meters-in-london/acorn_details.csv')
print(acorn_deets.dtypes)
acorn_deets.head()
holidays = pd.read_csv('/kaggle/input/smart-meters-in-london/uk_bank_holidays.csv')