# Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # visualising

import seaborn as sns # visualising
df = pd.read_csv('../input/hourly_irish_weather.csv')

df['date'] = pd.to_datetime(df['date'])
stations = df.groupby('station')['date'].min().keys()

counties = list(df.groupby('station')['county'].min())

min_date = list(df.groupby('station')['date'].min())

max_date = list(df.groupby('station')['date'].max())

min_temp = list(df.groupby('station')['temp'].min())

med_temp = list(df.groupby('station')['temp'].median())

max_temp = list(df.groupby('station')['temp'].max())

min_rain = list(df.groupby('station')['rain'].min())

med_rain = list(df.groupby('station')['rain'].median())

max_rain = list(df.groupby('station')['rain'].max())



summary_df = pd.DataFrame({

    'Station':stations,

    'County':counties,

    'Earliest Records':min_date,

    'Latest Records':max_date,

    'Minimum Temperature':min_temp,

    'Median Temperature':med_temp,

    'Maximum Temperature':max_temp,

    'Minimum Rain':min_rain,

    'Median Rain':med_rain,

    'Maximum Rain':max_rain

})



summary_df = summary_df.sort_values(by=['County'], ascending=True)
summary_df