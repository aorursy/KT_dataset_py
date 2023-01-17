# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mpl
files = []
files.append(pd.read_csv("../input/airline-2019/Jan2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/feb2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/mar2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/apr2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/may2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/june2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/jul2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/aug2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/sept2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/oct2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/nov2019/405557996_T_ONTIME_REPORTING.csv"))

files.append(pd.read_csv("../input/airline-2019/dec20219/405557996_T_ONTIME_REPORTING.csv"))

                         

df = pd.concat(files)
df
df.dtypes
#sort by flight date

df = df.sort_values(by=['FL_DATE'])
df
o_city = df['ORIGIN_CITY_NAME'].tolist()
new_o_city = []

for city in o_city:

    city = city.split(',')[0] #Split by comma and take the first part, which is the full city name

    new_o_city.append(city)
df['ORIGIN_CITY_NAME'] = new_o_city #Update ORIGIN_CITY_NAME column
df
#Do the same thing on dest city

d_city = df['DEST_CITY_NAME']

new_d_city = []

for city in d_city:

    city = city.split(',')[0]

    new_d_city.append(city)



df['DEST_CITY_NAME'] = new_d_city
df
df['Unnamed: 25'].value_counts(dropna = False)
df.drop('Unnamed: 25', axis = 1, inplace = True)
df
df['CANCELLED'].value_counts(dropna = False)
df['CANCELLATION_CODE'].value_counts(dropna = False)
df[df['CANCELLED'] == 0]['CANCELLATION_CODE'].value_counts(dropna = False)
df[df['CANCELLED'] == 1]['CANCELLATION_CODE'].value_counts(dropna = False)
df.drop('CANCELLED', axis = 1, inplace = True)
df.head()
df.shape
df['TOTAL_DELAY'] = df['CARRIER_DELAY'] + df['WEATHER_DELAY'] + df['NAS_DELAY'] + df['SECURITY_DELAY'] + df['LATE_AIRCRAFT_DELAY']
df_airlines = df.groupby('OP_CARRIER_AIRLINE_ID')['TOTAL_DELAY'].aggregate(np.sum).reset_index()

df_airlines.sort_values('TOTAL_DELAY', ascending = False)
df_airport = df.groupby('ORIGIN')['TOTAL_DELAY'].aggregate(np.sum).reset_index()

df_airport.sort_values('TOTAL_DELAY', ascending = False)
df[df['ORIGIN'] == 'ORD'][['ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_NM']].head()