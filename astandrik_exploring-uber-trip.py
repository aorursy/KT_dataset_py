import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

sns.set_style('whitegrid')



days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df_apr = pd.read_csv('../input/uber-raw-data-apr14.csv')

df_may = pd.read_csv('../input/uber-raw-data-may14.csv')

df_jun = pd.read_csv('../input/uber-raw-data-jun14.csv')

df_jul = pd.read_csv('../input/uber-raw-data-jul14.csv')

df_aug = pd.read_csv('../input/uber-raw-data-aug14.csv')

df_sep = pd.read_csv('../input/uber-raw-data-sep14.csv')

df = pd.concat([df_apr,df_may,df_jun,df_jul,df_aug,df_sep],axis=0)

df.shape
dict_base = pd.DataFrame({

    'Base': ['B02512','B02598','B02617','B02682','B02764','B02765','B02835','B02836'],

    'Name': ['Unter','Hinter','Weiter','Schmecken','Danach-NY','Grun','Dreist','Drinnen']

})
dict_base
df = pd.merge(df, dict_base, on='Base')
df.head()
import calendar

df['Date/Time'] = pd.to_datetime(df['Date/Time'],format='%m/%d/%Y %H:%M:%S')

df['Date'] = df['Date/Time'].dt.strftime('%Y-%m-%d')

df['Day'] = df['Date/Time'].dt.dayofweek

df['DayName'] = df['Date/Time'].dt.weekday_name

df['Month'] = df['Date/Time'].dt.month.apply(lambda x: calendar.month_abbr[x])

df['Year'] = df['Date/Time'].dt.year

df['Hour'] = df['Date/Time'].dt.hour

df['Minute'] = df['Date/Time'].dt.minute

df['Second'] = df['Date/Time'].dt.second
df.head(10)
sns.countplot('Month', data=df, palette='Set2')
fig, ax = plt.subplots(figsize=(15,5))

sns.countplot('Month', data=df, hue='Name', palette='Set2')
fig,ax = plt.subplots(figsize=(20,7))

sns.countplot('DayName', data=df, order=days, palette='husl')
total_trips = pd.DataFrame({

    'TotalTrips':df.groupby(['Base','Month','Day','DayName','Hour','Minute'])['Date/Time'].count()

}).reset_index()

total_trips = pd.merge(total_trips, dict_base, on='Base')

total_trips.shape
total_trips.head()
fig,ax = plt.subplots(figsize=(15,7))

ax.set_title('Average Trips per Day')

sns.pointplot(x='DayName',y='TotalTrips',data=total_trips,hue='Name', palette='husl')
fig,ax = plt.subplots(figsize=(17,6))

ax.set_title('Total Daily Trips by Month')

sns.pointplot('DayName','TotalTrips', hue='Month', data=total_trips, order=days,

         scale=2, hue_order=str.split('Apr May Jun Jul Aug Sep'), palette='husl', estimator=np.sum)
grid = sns.FacetGrid(total_trips, row='Name', col='Month', aspect=1, 

    col_order=str.split('Apr May Jun Jul Aug Sep'), palette='husl')

grid.map(sns.pointplot,'DayName','TotalTrips', order=days, scale=1, estimator=np.sum)

plt.tight_layout()
fig,ax = plt.subplots(figsize=(20,7))

sns.countplot('Hour', data=df)
fig,ax = plt.subplots(figsize=(15,10))

ax.set_title('Hourly Trips per Day')

sns.pointplot(x='Hour',y='TotalTrips',data=total_trips,hue='DayName',estimator=np.sum,

             palette='husl', scale=1.5)
daily_total = pd.DataFrame({

    'TotalTrips':df.groupby(['Date','Base'])['Date/Time'].count()

}).reset_index()

daily_total.index_col='Date'