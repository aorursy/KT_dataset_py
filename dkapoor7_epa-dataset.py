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

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

from bq_helper import BigQueryHelper



bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")

pollutants = ['o3', 'co', 'no2', 'so2', 'pm25_frm', 'pm10']

start_date = dt.date(2015, 1, 1)

end_date = dt.date(2015, 12, 31)
__print__ = print

def print(string):

    os.system(f'echo \"{string}\"')

    __print__(string)
def write_csv(title, df):

    print('Writing to ' + title)

    f = open(title, 'w+')

    df.to_csv(f)

    f.close()
for curr_date in pd.date_range(start_date, end_date):

    df = None



    for pollutant in pollutants:

        eda_query = """

        select 

            distinct concat(state_code, county_code) as combined_code,

            avg(aqi) as average_aqi,

            parameter_name,

            date_local

        from `bigquery-public-data.epa_historical_air_quality`.%s_daily_summary

        where date_local = date(%s, %s, %s)

        group by combined_code, parameter_name, date_local

        """ % (pollutant, curr_date.year, curr_date.month, curr_date.day)



        curr_df = bq_assistant.query_to_pandas(eda_query)



        #print(len(current_df))

        #print(current_df.head(10))

        #print('\n')

        

        #title = pollutant + '_' + str(curr_date.year) + '_' + str(curr_date.month) + '_' + str(curr_date.day) + '.csv'

        #write_csv(title, curr_df)

        



        if df is None:

            df = curr_df

        else:

            df = df.append(curr_df)



    title = 'all_pollutants_' + str(curr_date.year) + '_' + str(curr_date.month) + '_' + str(curr_date.day) + '.csv'

    write_csv(title, df)