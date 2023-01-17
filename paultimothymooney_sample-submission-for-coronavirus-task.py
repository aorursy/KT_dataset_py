# Import Python Packages

import pandas as pd

import numpy as np

import datetime as dt

import warnings

warnings.filterwarnings("ignore")



# Load the Data

nCoV_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

nCoV_data['Date'] = pd.to_datetime(nCoV_data['Date']).dt.normalize()

nCoV_data['ObservationDate'] = nCoV_data.Date.astype(str)

nCoV_data['Country/Region'] = nCoV_data['Country']

nCoV_data = nCoV_data[['ObservationDate','Province/State', 'Country/Region','Confirmed', 'Deaths', 'Recovered']]



# Filter Data for January 2020 Only

january = pd.date_range('2020-01', '2020-02', freq='D')

january = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04',

               '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',

               '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12',

               '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16',

               '2020-01-17', '2020-01-18', '2020-01-19', '2020-01-20',

               '2020-01-21', '2020-01-22', '2020-01-23', '2020-01-24',

               '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28',

               '2020-01-29', '2020-01-30', '2020-01-31']

nCoV_data_january_2020 = nCoV_data[nCoV_data['ObservationDate'].isin(january)]



# Double it and swap the date column for dates in March

df = nCoV_data_january_2020

df.Confirmed = df.Confirmed*2

df.Deaths = df.Deaths*2

df.Recovered = df.Recovered*2

nCoV_data_march_2020 = df

nCoV_data_march_2020['ObservationDate'] = nCoV_data_march_2020['ObservationDate'].str.replace("01",'03')

nCoV_data_march_2020.to_csv('/kaggle/working/my_submission.csv',index=False)
nCoV_data_march_2020.head(20)