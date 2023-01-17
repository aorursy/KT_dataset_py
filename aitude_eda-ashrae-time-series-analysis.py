import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

%matplotlib inline
DATA_PATH = '../input/ashrae-energy-prediction/'



train_df = pd.read_csv(DATA_PATH + 'train.csv',index_col='timestamp', parse_dates=True)

train_df['meter'] = pd.Categorical(train_df['meter']).rename_categories({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})
def seasonal_consumption(df,seasons):

    chart_rows = len(seasons)

    size = (16,chart_rows * 5)

    fig,ax = plt.subplots(chart_rows,1,figsize=size)

    for i,season in enumerate(seasons):

        season_name, season_months = season

        month_meters = df.loc[season_months].groupby('meter').mean()

        plt.subplot(chart_rows,1,i+1)

        plt.title('Energy Consumption in ' + str.capitalize(season_name))

        plt.bar(month_meters.index, month_meters)

        #plt.ylim(0,1e10)

        plt.ylabel('Energy Consumption (kWh)')

        plt.xlabel('Meter Type')

    

    fig.subplots_adjust(hspace=.3)

    plt.show()



def months_consumption(df,months):

    months_title = ['January','February','March','April','May','June','July','August','September','October','November','December']

    chart_rows = int(len(months)/2)

    size = (16,chart_rows * 5)

    fig,ax = plt.subplots(chart_rows,2,figsize=size)



    for month in months:

        month_meters = df.loc[month]

        plt.subplot(chart_rows,2,month)

        plt.title('Energy Consumption in '+ months_title[month-1])

        plt.bar(month_meters.index, month_meters)

        #plt.ylim(0,1e10)

        plt.ylabel('Energy Consumption (kWh)')

        plt.xlabel('Meter Type')

    

    fig.subplots_adjust(hspace=.3)

    plt.show()



def meter_type_consumption(df,meter_types):

    months_title = ['January','February','March','April','May','June','July','August','September','October','November','December']

    chart_rows = len(meter_types)

    size = (16,chart_rows * 5)

    fig,ax = plt.subplots(chart_rows,1,figsize=size)



    for i,meter in enumerate(meter_types):

        month_meters = df.loc[:,meter]

        plt.subplot(chart_rows,1,i+1)

        plt.title('Energy Consumption by '+ str.capitalize(meter))

        plt.bar(month_meters.index, month_meters)

        plt.ylabel('Energy Consumption (kWh)')

        plt.xlabel('Months')

        plt.xticks(np.arange(1,13),months_title, rotation=20)

    

    fig.subplots_adjust(hspace=.4)

    plt.show()
months_df = train_df.copy()

months_df['month'] = train_df.index

months_df['month'] = months_df['month'].dt.month

months_reading = months_df.groupby(['month','meter'])['meter_reading'].mean()
spring= ('spring',[3,4,5])

summer= ('summer',[6,7,8])

fall= ('fall',[9,10,11])

winter= ('winter',[1,2,12])



seasonal_consumption(months_reading,[spring,summer,fall,winter])
months_consumption(months_reading,[1,2,3,4,5,6,7,8,9,10,11,12])
meter_type_consumption(months_reading,['electricity','chilledwater','steam','hotwater'])
days_df = train_df.copy()

days_df['day'] = days_df.index

days_df['day'] = days_df['day'].dt.day

days_reading = days_df.groupby('day')['meter_reading'].mean()

plt.figure(figsize=(16,5))

plt.title('Day of the Month Observation')

plt.ylabel('Energy Consumption (kWh)')

plt.xlabel('Day')

plt.plot(days_reading)

plt.show()
weekday_df = train_df.copy()

weekday_df['weekday'] = weekday_df.index

weekday_df['weekday'] = weekday_df['weekday'].dt.weekday

weekday_reading = weekday_df.groupby('weekday')['meter_reading'].mean()

plt.figure(figsize=(15,5))

plt.title('Day of the Week Observation')

plt.ylabel('Energy Consumption (kWh)')

plt.xlabel('Day')

plt.plot(weekday_reading)

plt.show()
hour_df =train_df.copy()

hour_df['hour'] = hour_df.index

hour_df['hour'] = hour_df['hour'].dt.hour

hour_reading = hour_df.groupby('hour')['meter_reading'].mean()

plt.figure(figsize=(15,5))

plt.title('Hour of the Day Observation')

plt.ylabel('Energy Consumption (kWh)')

plt.xlabel('Hour')

plt.plot(hour_reading)

plt.show()