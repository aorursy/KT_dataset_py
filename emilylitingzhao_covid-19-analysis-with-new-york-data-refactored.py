#import packages
import numpy as np
import pandas as pd
import requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
from matplotlib import pyplot as plt
# load data
us_data=pd.read_csv('confirmed-covid-19-cases-in-us-by-state-and-county.csv')
us_data.head()
# cleaning data
us_data.isnull().sum()
columns=['lat','long','geometry']
us_data.drop(columns, axis='columns',inplace=True)
us_data.isnull().sum()
# check the data type of date column, then convert it to datetime type
us_data.info()
us_data['date']=pd.to_datetime(us_data['date'],format="%Y-%m-%d")
us_data.info()
# find out the time period 
us_data['date'].sort_values(ascending=True)
us_data_ny=us_data[us_data['state_name']=='NY']
#us_data_ny.set_index('date', inplace=True)
counties=['Albany County', 'Allegany County','Bronx County',
         'Cattaraugus County','Ontario County','Queens County',
         'Rockland County','Westchester County']
for county in counties:
    df = us_data_ny[us_data_ny['county_name']==county]
    plt.plot(df.date, df.confirmed)
    title= f'Confirmed cases in {county}'
    print(title)
    plt.xlabel('date')
    plt.xticks(rotation=90)
    plt.ylabel('confirmed cases')
    plt.show()
#calculate the increasing rate of new cases
def growth_rate(county):    
    c_df=us_data_ny[us_data_ny['county_name']==county].reset_index()
    last_day = c_df['confirmed'].shift(1,axis=0)
    last_day = pd.DataFrame(last_day).reset_index()
    c_df['confirmed']
    c_df['ratio']=(c_df['confirmed']/last_day['confirmed'])-1
    c_df['ratio']=c_df['ratio'].fillna(0)
    c_df.set_index('date', inplace=True)
    plt.plot(c_df.index, c_df.ratio)
    print(f'New cases growth rate in {county}')
    plt.xlabel('date')
    plt.xticks(rotation=90)
    plt.ylabel('new cases growth rate')
    plt.show()
# call the method on the counties list
for county in counties:
    growth_rate(county)