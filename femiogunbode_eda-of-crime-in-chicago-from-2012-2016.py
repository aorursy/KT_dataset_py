import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
first_set = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')

# Drop rows that have 2017 as the year.

crimes = pd.DataFrame(first_set[first_set['Year'] != 2017])

crimes.head()
# Group by Crime type and calculate count

crime_count = pd.DataFrame(crimes.groupby('Primary Type').size().sort_values(ascending=False).rename('Count').reset_index())

crime_count.head()
crime_count.shape
# Plot top 10 crimes on a barplot

crime_count[:10].plot(x='Primary Type',y='Count',kind='bar')
# Group by Crime Location and calculate count

crime_location = pd.DataFrame(crimes.groupby('Location Description').size().sort_values(ascending=False).rename('Count').reset_index())

crime_location.head()
crime_location.shape
# Plot top 10 crime location on a barplot

crime_location[:10].plot(x='Location Description',y='Count',kind='bar')
import calendar

from datetime import datetime

crimes['NewDate'] =  crimes['Date'].apply(lambda x: datetime.strptime(x.split()[0],'%m/%d/%Y'))
crimes['MonthNo'] = crimes['Date'].apply(lambda x: str(x.split()[0].split('/')[0]))
monthDict = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}

crimes['Month'] = crimes['MonthNo'].apply(lambda x: monthDict[x])

crimes.head()
crime_activity_plot = pd.DataFrame(crimes.groupby(['Month','Year']).size().sort_values(ascending=False).rename('Count').reset_index())

crime_activity_plot.head()
crime_activity_plot_2012_2016 = crime_activity_plot.pivot_table(values='Count',index='Month',columns='Year')
sns.heatmap(crime_activity_plot_2012_2016)
sns.clustermap(crime_activity_plot_2012_2016,cmap='coolwarm')
arrest_yearly = crimes[['Year','Arrest','Month']]

arrest_yearly_new = arrest_yearly[arrest_yearly['Arrest'] == True]

arrest_yearly_plot = pd.DataFrame(arrest_yearly_new.groupby(['Month','Year']).size().sort_values(ascending=False).rename('Count').reset_index())

arrest_yearly_plot.head()
arrest_yearly_matrix = arrest_yearly_plot.pivot_table(values='Count',index='Month',columns='Year')
sns.heatmap(arrest_yearly_matrix)
crime_activity = pd.DataFrame(crimes.groupby('Year').size().rename('Count').reset_index())

crime_activity

arrest = crimes[['Year','Arrest']]

arrest_new = arrest[arrest['Arrest'] == True]

arrest_activity = pd.DataFrame(arrest_new.groupby('Year').size().rename('Count').reset_index())

arrest_activity
import matplotlib.ticker as ticker

x=['2012','2013','2014','2015','2016']

y=crime_activity['Count']

z=arrest_activity['Count']

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

ax.plot(x,y,label='Crime Activity')

ax.plot(x,z,label='Arrests')

ax.set_ylabel("COUNT")

ax.set_xlabel("YEAR")

ax.set_title("Crime Activity VS Arrests from 2012 - 2016")

ax.legend()