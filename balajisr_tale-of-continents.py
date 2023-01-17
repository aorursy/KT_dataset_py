import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os



temp = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')

temp = temp[temp['AvgTemperature'] != -99]
sns.distplot(temp['AvgTemperature'],kde=True)
temp['AvgTemperature'].mean()
temp[temp['AvgTemperature'] == max(temp['AvgTemperature'])]
temp[temp['AvgTemperature'] == min(temp['AvgTemperature'])]
plt.figure(figsize=(8,3))

region_stats = temp.groupby('Region')['AvgTemperature'].agg(mean_temp='mean',std_temp = 'std',min_temp = 'min',max_temp = 'max',median_temp = 'median').reset_index().sort_values('mean_temp',ascending=False)

sns.barplot(x='mean_temp',y='Region',data=region_stats)

plt.xlabel("Mean Temperature",fontsize=12)

plt.ylabel('Continents',fontsize=12)

plt.title("Mean Temperature by Continents",fontsize=16)
region_stats['cv'] = region_stats['std_temp'] / region_stats['mean_temp']

region_stats = region_stats.sort_values('cv',ascending=False)

plt.figure(figsize=(8,3))

sns.barplot(x='cv',y='Region',data=region_stats)

plt.xlabel("Coefficent of Variation - Temperature",fontsize=12)

plt.ylabel('Continents',fontsize=12)

plt.title("Coefficent of Variation by Continents",fontsize=16)
plt.figure(figsize=(12,6))

city_stats = temp.groupby('City')['AvgTemperature'].agg(mean_temp='mean',std_temp = 'std',min_temp = 'min',max_temp = 'max',median_temp = 'median').reset_index().sort_values('mean_temp',ascending=False).head(20)

sns.barplot(x='mean_temp',y='City',data=city_stats)

plt.xlabel("Mean Temperature",fontsize=12)

plt.ylabel('City',fontsize=12)

plt.title("Top 20 Hottest Cities",fontsize=16)
plt.figure(figsize=(12,6))

city_stats = temp.groupby('City')['AvgTemperature'].agg(mean_temp='mean',std_temp = 'std',min_temp = 'min',max_temp = 'max',median_temp = 'median').reset_index().sort_values('mean_temp').head(20)

sns.barplot(x='mean_temp',y='City',data=city_stats)

plt.xlabel("Mean Temperature",fontsize=12)

plt.ylabel('City',fontsize=12)

plt.title("Top 20 Coldest Cities",fontsize=16)
city_stats['cv'] = city_stats['std_temp'] / city_stats['mean_temp']

city_stats = city_stats.sort_values('cv',ascending=False).head(20)

plt.figure(figsize=(12,6))

sns.barplot(x='cv',y='City',data=city_stats)

plt.xlabel("Coefficient of Variation - Temperature",fontsize=12)

plt.ylabel('City',fontsize=12)

plt.title("Coefficent of Variation by Cities",fontsize=16)
city_stats[city_stats['City']=='Fairbanks']
America = temp[temp['Region']=='North America']

plt.figure(figsize=(10,6))

america_stats = America.groupby('Month')['AvgTemperature'].agg(mean_temp='mean').reset_index()

sns.barplot(x='Month',y='mean_temp',data=america_stats)

plt.xlabel("Month",fontsize=12)

plt.ylabel('Temperature',fontsize=12)

plt.title("Mean Temperature by Month",fontsize=16)
Europe = temp[temp['Region']=='Europe']

plt.figure(figsize=(10,6))

europe_stats = Europe.groupby('Month')['AvgTemperature'].agg(mean_temp='mean').reset_index()

sns.barplot(x='Month',y='mean_temp',data=europe_stats)

plt.xlabel("Month",fontsize=12)

plt.ylabel('Temperature',fontsize=12)

plt.title("Mean Temperature by Month",fontsize=16)
Asia = temp[temp['Region']=='Asia']

plt.figure(figsize=(10,6))

asia_stats = Asia.groupby('Month')['AvgTemperature'].agg(mean_temp='mean').reset_index()

sns.barplot(x='Month',y='mean_temp',data=asia_stats)

plt.xlabel("Month",fontsize=12)

plt.ylabel('Temperature',fontsize=12)

plt.title("Mean Temperature by Month",fontsize=16)
Aus = temp[temp['Region']=='Australia/South Pacific']

plt.figure(figsize=(10,6))

aus_stats = Aus.groupby('Month')['AvgTemperature'].agg(mean_temp='mean').reset_index()

sns.barplot(x='Month',y='mean_temp',data=aus_stats)

plt.xlabel("Month",fontsize=12)

plt.ylabel('Temperature',fontsize=12)

plt.title("Mean Temperature by Month",fontsize=16)
Africa = temp[temp['Region']=='Africa']

plt.figure(figsize=(10,6))

africa_stats = Africa.groupby('Month')['AvgTemperature'].agg(mean_temp='mean').reset_index()

sns.barplot(x='Month',y='mean_temp',data=africa_stats)

plt.xlabel("Month",fontsize=12)

plt.ylabel('Temperature',fontsize=12)

plt.title("Mean Temperature by Month",fontsize=16)
Middle_East = temp[temp['Region']=='Middle East']

plt.figure(figsize=(10,6))

middle_east_stats = Middle_East.groupby('Month')['AvgTemperature'].agg(mean_temp='mean').reset_index()

sns.barplot(x='Month',y='mean_temp',data=middle_east_stats)

plt.xlabel("Month",fontsize=12)

plt.ylabel('Temperature',fontsize=12)

plt.title("Mean Temperature by Month",fontsize=16)
India = temp[temp['Country']=='India']

plt.figure(figsize=(10,6))

India_stats = India.groupby('Month')['AvgTemperature'].agg(mean_temp='mean').reset_index()

sns.barplot(x='Month',y='mean_temp',data=India_stats)

plt.xlabel("Month",fontsize=12)

plt.ylabel('Temperature',fontsize=12)

plt.title("Mean Temperature by Month",fontsize=16)
plt.figure(figsize=(12,6))

temp.head()

year_stats = temp.groupby(['Year','Region'])['AvgTemperature'].agg(mean_temp='mean').reset_index()

year_stats.head()

sns.lineplot(x='Year',y='mean_temp',hue='Region',data=year_stats)

plt.xlabel("Year",fontsize=12)

plt.ylabel('Temperature',fontsize=12)

plt.title("Mean Temperature by Year",fontsize=16)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)