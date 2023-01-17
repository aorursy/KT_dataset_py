# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from google.cloud import bigquery
%%time

client = bigquery.Client()

dataset_ref = client.dataset("noaa_gsod", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))



table_ref = dataset_ref.table("stations")

table = client.get_table(table_ref)

stations_df = client.list_rows(table).to_dataframe()



table_ref = dataset_ref.table("gsod2020")

table = client.get_table(table_ref)

twenty_twenty_df = client.list_rows(table).to_dataframe()



stations_df['STN'] = stations_df['usaf'] + '-' + stations_df['wban']

twenty_twenty_df['STN'] = twenty_twenty_df['stn'] + '-' + twenty_twenty_df['wban']



cols_1 = ['STN', 'mo', 'da', 'temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']

cols_2 = ['STN', 'country', 'state', 'call', 'lat', 'lon', 'elev']

weather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index('STN'), on='STN')



weather_df.tail(10)
weather_df.to_csv('Weather_Week2.csv',index=False)