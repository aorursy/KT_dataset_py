# Imports

%config InlineBackend.figure_format = 'retina'

%matplotlib inline



import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

import pandas as pd



import seaborn as sns

sns.set_style('white')
df_aircode1 = pd.DataFrame.from_csv('../input/L_AIRPORT.csv')

df_aircode2 = pd.DataFrame.from_csv('../input/L_AIRPORT_ID.csv')
# Format the airport codes

df_aircode1 = df_aircode1.reset_index()

df_aircode2 = df_aircode2.reset_index()

df_aircodes = pd.merge(df_aircode1,df_aircode2,on='Description')

aircode_dict = dict(zip(df_aircodes['Code_y'].astype(str),df_aircodes['Code_x']))
# Load data

df_fl = pd.io.parsers.read_csv('../input/flights.csv')



# Make sure all Origin and departing airports are strings

df_fl['ORIGIN_AIRPORT'] = df_fl['ORIGIN_AIRPORT'].values.astype(str)

df_fl['DESTINATION_AIRPORT'] = df_fl['DESTINATION_AIRPORT'].values.astype(str)



N_flights = len(df_fl)

for i in range(N_flights):

    if i % 100000 == 0:

        print(i)

    if len(df_fl['ORIGIN_AIRPORT'][i]) != 3:

        to_replace = df_fl['ORIGIN_AIRPORT'][i]

        value = aircode_dict[df_fl['ORIGIN_AIRPORT'][i]]

        df_fl = df_fl.replace(to_replace, value)

        print('replaced',to_replace,'with',value)

    elif len(df_fl['DESTINATION_AIRPORT'][i]) != 3:

        to_replace = df_fl['DESTINATION_AIRPORT'][i]

        value = aircode_dict[df_fl['DESTINATION_AIRPORT'][i]]

        df_fl = df_fl.replace(to_replace, value)

        print('replaced',to_replace,'with',value)