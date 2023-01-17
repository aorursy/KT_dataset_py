# IMPORTS

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()    

%matplotlib inline
# LOAD THE DATA

data = pd.read_csv('../input/EarthQuakes in Greece.csv')
data.info()
data.describe()
data[data['MAGNITUDE (Richter)'] == 8]
data.head()
columns = ['year', 'month', 'day', 'hours', 'minutes', 'LAT', 'LON', 'richter']



data.columns = columns



data['date'] = pd.to_datetime(data[['year','month','day','hours','minutes']])



data.drop(['year', 'month', 'day', 'hours', 'minutes'], axis=1, inplace=True)
data.head()
# PLOTS & EDA
data['date'].hist(bins=30)

plt.yscale('log')

plt.xlabel('Year')

plt.tight_layout()
data['richter'].hist(bins=8)

plt.yscale('log')

plt.xlabel('Magnitude (Richter)')

plt.tight_layout()
sns.boxplot(x='richter', data=data, )
data[data['richter'] == 0]
year = data['date'].apply(lambda x: x.year) #we separate the year from the date

month = data['date'].apply(lambda x: x.month) #we separate the month from the date

hour = data['date'].apply(lambda x: x.hour) #we separate the hours from the date



pivot_year = pd.pivot_table(data, values='richter', index=year)#we group the data for each year in a pivot table

pivot_month = pd.pivot_table(data, values='richter', index=month)#we group the data for each month in a pivot table

pivot_hour = pd.pivot_table(data, values='richter', index=hour)#we group the data for each hour in a pivot table
f, axes = plt.subplots(3, 2, figsize=(10, 10))

sns.heatmap(pivot_year,yticklabels='auto', cmap='viridis', ax=axes[0][0])

sns.heatmap(pivot_month,yticklabels='auto', cmap='viridis', ax=axes[1][0])

sns.heatmap(pivot_hour,yticklabels='auto', cmap='viridis', ax=axes[2][0])



pivot_year.plot(ax=axes[0][1])

pivot_month.plot(ax=axes[1][1])

pivot_hour.plot(ax=axes[2][1])





plt.tight_layout()

sns.scatterplot(x='LAT', y='LON', data=data)
sns.scatterplot(x='LAT', y='LON', data=data)

plt.axvline(33.957559, color='r')

plt.axvline(44.108926, color='r')

plt.axhline(18.17496, color='r')

plt.axhline(32.061679,color='r')

plt.tight_layout()
import folium
m = folium.Map([data['LAT'].mean(), data['LON'].mean()],zoom_start=6)

m
def make_map(clm):

    lat = clm['LAT']

    lon = clm['LON']

    mag = clm['richter']

    year = clm['date'].year





    folium.Circle(

        radius=2000 * mag,

        location=[lat, lon],

        popup="Year: " + str(year) +" " + "Magnitude: " + " " +str(mag) + " richter",

        color='crimson',

        fill=False,

    ).add_to(m)

filter_richter = data['richter'] >= 7

filtered_data = data[filter_richter]



m = folium.Map([data['LAT'].mean(), data['LON'].mean()], zoom_start=6)

_ = filtered_data.apply(lambda x:make_map(x), axis=1)



m