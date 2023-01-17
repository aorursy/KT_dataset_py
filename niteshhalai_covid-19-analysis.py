# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime, timedelta

import matplotlib.dates as mdates

import plotly.graph_objects as go

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)





sns.set(style="whitegrid", color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
world_data = pd.read_csv('/kaggle/input/covid19-by-country-with-government-response/covid19_by_country.csv')

world_data
world_data['Date'] = pd.to_datetime(world_data['Date'])

world_data['active'] = world_data['confirmed'] - world_data['deaths'] - world_data['recoveries']

world_data['closed cases'] = world_data['deaths'] + world_data['recoveries']

world_data
len(world_data['Country'].unique())
world_last_date = pd.DataFrame(world_data.tail(1)['Date']).iloc[0]['Date']

pd.DataFrame(world_data[world_data['Date']==world_last_date].head(10000).groupby(by ='Date').sum()[['confirmed', 'active', 'closed cases', 'deaths', 'recoveries']])
worlds_data = pd.DataFrame(world_data.groupby(by ='Date').sum()[['confirmed', 'active', 'closed cases', 'deaths', 'recoveries']])



fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(25,7), sharex=True, sharey=True)



ax[0].title.set_text('World: closed cases vs confirmed cases vs recovered cases')

ax[0].plot(worlds_data.index, worlds_data['confirmed'], label='confirmed cases')

ax[0].plot(worlds_data.index, worlds_data['active'], label='active cases')

ax[0].plot(worlds_data.index, worlds_data['closed cases'], label='closed cases')

ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=10))

ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

ax[0].tick_params(labelrotation=25)

ax[0].legend(fontsize='x-large', fancybox=True)



ax[1].title.set_text('World: breakdown of closed cases')

ax[1].plot(worlds_data.index, worlds_data['recoveries'], label='recoveries')

ax[1].plot(worlds_data.index, worlds_data['deaths'], label='deaths')

ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=10))

ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

ax[1].tick_params(rotation=45)

ax[1].legend(fontsize='x-large', fancybox=True)





plt.show()
temp5 = pd.DataFrame(world_data[world_data['Date']==world_last_date].head(10000).groupby(by =['Date', 'Country']).sum()[['confirmed', 'active', 'recoveries', 'deaths', 'closed cases']])

temp5 = temp5[temp5['active']==0]

#temp5

temp5 = temp5.reset_index()

zero_case_countries = list(temp5['Country'].unique())

columns = ['Country', 'confirmed', 'active', 'recoveries', 'deaths', 'closed cases']



fig = go.Figure(data=[go.Table(

    header=dict(values=list(columns),

                fill_color='paleturquoise',

                align='left'),

    cells=dict(values=[temp5['Country'], temp5['confirmed'], temp5['active'], temp5['recoveries'], temp5['deaths'], temp5['closed cases']],

               fill_color='white',

               align='left'))

])





fig.show()





temp3 = pd.DataFrame(world_data[world_data['Date']==world_last_date][['Country', 'confirmed', 'active', 'recoveries', 'deaths', 'closed cases']]).sort_values(by='confirmed',ascending=False)

plt.figure(figsize=(15,7))

plt.title('Zero active cases countires', fontsize='xx-large')



data = temp3[temp3['Country'].isin(zero_case_countries)]



ax = sns.barplot(data=data, y=data['Country'], x=data['confirmed'], color='orange')

ax = sns.barplot(data=data, y=data['Country'], x=data['closed cases'], color='green')

ax = sns.barplot(data=data, y=data['Country'], x=data['deaths'], color='red')



ax.set(xlabel='')

            

plt.show()
temp3 = pd.DataFrame(world_data[world_data['Date']==world_last_date][['Country', 'confirmed', 'active', 'recoveries', 'deaths', 'closed cases']]).sort_values(by='confirmed',ascending=False)

plt.figure(figsize=(15,7))

plt.title('Confirmed cases country wise', fontsize='xx-large')



data=temp3.head(25)

pal = sns.light_palette("red", reverse=True)



ax = sns.barplot(data=data, y=data['Country'], x=data['confirmed'], color='orange')

ax = sns.barplot(data=data, y=data['Country'], x=data['closed cases'], color='green')

ax = sns.barplot(data=data, y=data['Country'], x=data['deaths'], color='red')



ax.set(xlabel='')

            

plt.show()
temp3 = pd.DataFrame(world_data[world_data['Date']==world_last_date][['Country', 'active']]).sort_values(by='active',ascending=False)

plt.figure(figsize=(15,7))

plt.title('Active cases country wise', fontsize='xx-large')

data3=temp3.head(25)

pal = sns.light_palette("orange", len(temp3.head(50)), reverse=True)

rank = data3['active'].argsort().argsort()

sns.barplot(data=data3, y=data3['Country'], x=data3['active'], palette=np.array(pal[::-1])[rank])



plt.show()


temp3 = pd.DataFrame(world_data[world_data['Date']==world_last_date][['Country', 'recoveries']]).sort_values(by='recoveries',ascending=False)

plt.figure(figsize=(15,7))

plt.title('Recoveries country wise', fontsize='xx-large')

data3=temp3.head(25)

pal = sns.light_palette("green", len(temp3.head(50)), reverse=True)

rank = data3['recoveries'].argsort().argsort()

sns.barplot(data=data3, y=data3['Country'], x=data3['recoveries'], palette=np.array(pal[::-1])[rank])



plt.show()
temp3 = pd.DataFrame(world_data[world_data['Date']==world_last_date][['Country', 'deaths']]).sort_values(by='deaths',ascending=False)

plt.figure(figsize=(15,7))

plt.title('Deaths country wise', fontsize='xx-large')

data3=temp3.head(25)

pal = sns.light_palette("red", len(temp3.head(50)), reverse=True)

rank = data3['deaths'].argsort().argsort()

sns.barplot(data=data3, y=data3['Country'], x=data3['deaths'], palette=np.array(pal[::-1])[rank])



plt.show()
def plot(country):

    

    

    temp3 = pd.DataFrame(world_data[world_data['Date']==world_last_date][['Country', 'confirmed', 'active', 'recoveries', 'deaths', 'closed cases']]).sort_values(by='confirmed',ascending=False)

    plt.figure(figsize=(25,1))

    plt.title(country+': current status', fontsize='xx-large')



    data=temp3[temp3['Country']==country]

    pal = sns.light_palette("red", reverse=True)



    ax = sns.barplot(data=data, y=data['Country'], x=data['confirmed'], color='orange')

    ax = sns.barplot(data=data, y=data['Country'], x=data['closed cases'], color='green')

    ax = sns.barplot(data=data, y=data['Country'], x=data['deaths'], color='red')



    ax.set(xlabel='')

            

    plt.show()

    

    country_data = world_data[(world_data['Country']==country)&(world_data['confirmed']>0)]

    country_data.set_index(keys='Date', inplace=True)

    country_data.index = pd.to_datetime(country_data.index)

    

    fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(25,7), sharex=True, sharey=True)

    

    ax[0].title.set_text(country + ': closed cases vs confirmed cases vs recovered cases')

    ax[0].plot(country_data.index,country_data['confirmed'], label='confirmed cases')

    ax[0].plot(country_data.index, country_data['active'], label='active cases')

    ax[0].plot(country_data.index, country_data['closed cases'], label='closed cases')

    ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=10))

    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    ax[0].tick_params(labelrotation=45)

    ax[0].legend(fontsize='xx-large', fancybox=True)

    

    

    ax[1].title.set_text(country + ': breakdown of closed cases')

    ax[1].plot(country_data.index, country_data['recoveries'], label='recoveries')

    ax[1].plot(country_data.index, country_data['deaths'], label='deaths')

    ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=10))

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    ax[1].tick_params(labelrotation=45)

    ax[1].legend(fontsize='xx-large', fancybox=True)

    

    plt.show()

plot('India')
india_data = pd.read_csv('/kaggle/input/covid19-corona-virus-india-dataset/complete.csv')
last_date = pd.DataFrame(india_data.tail(1)['Date']).iloc[0]['Date']

#pd.DataFrame(india_data[india_data['Date']==last_date].head(100).groupby(by = 'Date').sum()['Total Confirmed cases'])
temp2 = pd.DataFrame(india_data[india_data['Date']==last_date][['Name of State / UT', 'Total Confirmed cases']]).sort_values(by='Total Confirmed cases',ascending=False)

plt.figure(figsize=(15,7))

sns.barplot(data=temp2.head(20), x='Total Confirmed cases', y='Name of State / UT')

plt.show()
india_data.set_index(keys='Date', inplace=True)

india_data.index = pd.to_datetime(india_data.index)
fig, ax=plt.subplots(figsize=(20,7))

plt.title('Statewise daily rise in confirmed cases', fontsize='xx-large')



for state in list(temp2.head(10)['Name of State / UT']):

    ax.plot(india_data[(india_data['Name of State / UT']==state)].index, india_data.loc[(india_data['Name of State / UT']==state)]['Total Confirmed cases'], label=state)



ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

plt.xticks(rotation=45)

#plt.legend(fontsize='large', fancybox=True)

plt.show()
patients_data = pd.read_csv('/kaggle/input/covid19-corona-virus-india-dataset/patients_data.csv')

patients_data.head()
#status_change_days = patients_data[['date_announced','status_change_date','current_status']]#

#status_change_days['date_announced'] = status_change_days['date_announced'].astype('str')

#status_change_days.dropna(axis=0, how='any', inplace=True)

#status_change_days['status_change_date'] = status_change_days['status_change_date'].apply(lambda _: datetime.strptime(_,'%d/%m/%Y'))

#status_change_days['date_announced'] = status_change_days['date_announced'].apply(lambda _: datetime.strptime(_,'%d/%m/%Y'))

#status_change_days['diff_days'] = (status_change_days['status_change_date'] - status_change_days['date_announced'])/np.timedelta64(1,'D')

#plt.figure(figsize=(15,7))

#print('Records available: ', status_change_days.shape[0])

#sns.violinplot(data=status_change_days[status_change_days['current_status']!='Hospitalised'],x='current_status', y='diff_days')

#plt.show()
temp3 = pd.DataFrame(world_data[world_data['Date']==world_last_date][['Country', 'confirmed']]).sort_values(by='confirmed',ascending=False)

countries = list(temp3['Country'].unique())



for country in countries:

    plot(country)