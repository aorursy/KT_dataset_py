import warnings

import itertools

import numpy as np

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

import pandas as pd

import statsmodels.api as sm

import matplotlib
def read_pickle_file(file):

    pickle_data = pd.read_pickle(file)

    return pickle_data
df_accidents_2009_2011 = pd.read_csv('../input/traffic-accidents/accidents_2009_to_2011.csv')

df_accidents_2012_2014 = pd.read_csv('../input/traffic-accidents/accidents_2012_to_2014.csv')

df_accidents = pd.concat([df_accidents_2012_2014, df_accidents_2009_2011])
df_accidents['Date'] = pd.to_datetime(df_accidents['Date'])
df_accidents.Date
df_accidents['Date'].min(), df_accidents['Date'].max()
cols = ['Accident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR',

       'Longitude', 'Latitude', 'Police_Force', 'Accident_Severity',

       'Number_of_Vehicles', 'Day_of_Week',

       'Time', 'Local_Authority_(District)', 'Local_Authority_(Highway)',

       '1st_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit',

       'Junction_Detail', 'Junction_Control', '2nd_Road_Class',

       '2nd_Road_Number', 'Pedestrian_Crossing-Human_Control',

       'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',

       'Weather_Conditions', 'Road_Surface_Conditions',

       'Special_Conditions_at_Site', 'Carriageway_Hazards',

       'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident',

       'LSOA_of_Accident_Location', 'Year']

df_accidents.drop(cols, axis=1, inplace=True)

df_accidents = df_accidents.sort_values('Date')
df_accidents = df_accidents.groupby('Date')['Number_of_Casualties'].sum().reset_index()

df_accidents = df_accidents.rename(columns={'Date': 'ds', 'Number_of_Casualties': 'y'})

df_accidents.set_index('ds')
df_accidents.dtypes
df_accidents['y'].plot(figsize=(15, 6))

plt.show()
import fbprophet

m = fbprophet.Prophet()

m.fit(df_accidents)
future = m.make_future_dataframe(periods=1825)
forecast = m.predict(future)
m.plot(forecast);
m.plot_components(forecast);