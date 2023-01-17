#Importing all libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

import datetime

%matplotlib inline
#Import the dataset

nyphil = pd.read_csv('../input/ny_phil.csv')
nyphil.shape
nyphil.info()
nyphil.head()
nyphil.isnull().sum()
#Convert the test to date format

nyphil['Date2'] = pd.to_datetime(nyphil['Date'],infer_datetime_format=True)

nyphil['Date3'] = nyphil['Date2'].dt.date

nyphil['year'] = nyphil['Date2'].dt.year

nyphil['month'] = nyphil['Date2'].dt.month
nyphil.columns
nyphil['eventType'].unique()
year_events = nyphil.groupby('year')['programID'].nunique().reset_index(name="count")
year_events.plot(x='year', figsize=[20,8], title='Number of Events by Year')
events = nyphil[nyphil['year']==2016].groupby('eventType')['programID'].nunique().reset_index(name="count").sort_values(by='count', ascending=True)

events.plot(kind='barh', x='eventType', y='count', figsize=[10,10], fontsize=15, title= 'Number of Events by EventType')
conductor_events = nyphil[nyphil['year']==2016].groupby('conductorName')['programID'].nunique().reset_index(name="count").sort_values(by='count', ascending=True)
conductor_events.plot(kind='barh', x='conductorName', y='count', figsize=[10,10])
venue_events = nyphil[nyphil['year']==2016].groupby('Venue')['programID'].nunique().reset_index(name="count").sort_values(by='count', ascending=True)
venue_events.plot(kind='barh', x='Venue', y='count', figsize=[10,10], title='Popular Venues in 2016')
composer_count = nyphil.groupby('composerName')['programID'].nunique().reset_index(name="count").sort_values(by='count', ascending=True)
composer_count[composer_count['count']>500].plot(kind='barh', x='composerName', y='count', figsize=[10,10], 

                                                 title='Most popular composers across time')