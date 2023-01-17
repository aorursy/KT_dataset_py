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

from bq_helper import BigQueryHelper



bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
pollutants = ['o3', 'co', 'no2', 'so2', 'pm25_frm', 'pm10']

earliest_year = '2000'

rows = 10
# get the number of rows of data available per county



for pollutant in pollutants:

    eda_query = """

    select 

        county_code,

        count(*) as hits

    from `bigquery-public-data.epa_historical_air_quality`.%s_daily_summary

    where poc = 1

        and extract(year from date_local) > %s

    group by county_code

    order by count(*) desc

    """ % (pollutant, earliest_year)

    

    df = bq_assistant.query_to_pandas(eda_query)

    

    print(pollutant + ' Results:')

    print(df.head(rows))

    print('\n')
# get the number of rows of data available per state



for pollutant in pollutants:

    eda_query = """

    select 

        state_name,

        count(*) as hits

    from `bigquery-public-data.epa_historical_air_quality`.%s_daily_summary

    where poc = 1

        and extract(year from date_local) > %s

    group by state_name

    order by count(*) desc

    """ % (pollutant, earliest_year)

    

    df = bq_assistant.query_to_pandas(eda_query)

    

    print(pollutant + ' Results:')

    print(df.head(rows))

    print('\n')
# get the number of rows of data available per city



for pollutant in pollutants:

    eda_query = """

    select 

        city_name,

        count(*) as hits

    from `bigquery-public-data.epa_historical_air_quality`.%s_daily_summary

    where poc = 1

        and extract(year from date_local) > %s

    group by city_name

    order by count(*) desc

    """ % (pollutant, earliest_year)

    

    df = bq_assistant.query_to_pandas(eda_query)

    

    print(pollutant + ' Results:')

    print(df.head(rows))

    print('\n')
# get the number of rows of data available per state+county



for pollutant in pollutants:

    eda_query = """

    select 

        concat(state_code, county_code) as combined_code,

        count(*) as hits

    from `bigquery-public-data.epa_historical_air_quality`.%s_daily_summary

    where poc = 1

        and extract(year from date_local) > %s

    group by state_code, county_code

    order by count(*) desc

    """ % (pollutant, earliest_year)

    

    df = bq_assistant.query_to_pandas(eda_query)

    

    print(pollutant + ' Results:')

    print(df.head(rows))

    print('\n')

    

    df.to_csv('state_county_row_count_' + pollutant + '.csv', index = False)
# get the avergae aqi of data available per state+county on 2017-03-03



for pollutant in pollutants:

    eda_query = """

    select 

        concat(state_code, county_code) as combined_code,

        avg(aqi) as average_aqi

    from `bigquery-public-data.epa_historical_air_quality`.%s_daily_summary

    where poc = 1

        and extract(year from date_local) = 2017-03-03

    group by state_code, county_code

    """ % (pollutant)

    

    df = bq_assistant.query_to_pandas(eda_query)

    print(len(df))

    print(pollutant + ' Results:')

    print(df.head(rows))

    print('\n')

    

    df.to_csv('state_county_2017_03_03_' + pollutant + '.csv', index = False)
df = None



for pollutant in pollutants:

    eda_query = """

    select 

        concat(state_code, county_code) as combined_code,

        avg(aqi) as average_aqi,

        parameter_name

    from `bigquery-public-data.epa_historical_air_quality`.%s_daily_summary

    where date_local = date(2017, 3, 3)

    group by state_code, county_code, parameter_name

    """ % (pollutant)

    

    current_df = bq_assistant.query_to_pandas(eda_query)

    print(len(current_df))

    print(current_df.head(rows))

    print('\n')

    

    if df is None:

        df = current_df

    else:

        df.append(current_df)

    

df.to_csv('state_county_2011_12_24.csv')