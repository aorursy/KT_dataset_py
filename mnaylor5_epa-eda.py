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
eda_query = """

select 

    state_name

    ,county_name

    ,city_name

    ,latitude

    ,longitude

    ,site_num

    ,date_local

    ,aqi as o3_aqi

    ,arithmetic_mean as o3_mean

from `bigquery-public-data.epa_historical_air_quality`.o3_daily_summary

where poc = 1

    and extract(year from date_local) > 2015

    and state_name in ('Tennessee', 'Georgia')

    and county_name in ('Shelby','Davidson','Fulton')

order by state_name, county_name, date_local

"""



df = bq_assistant.query_to_pandas(eda_query)

df.head()
group_df = df.groupby(['state_name','county_name','city_name', 'date_local'],as_index=False).mean()

group_df['date'] = pd.to_datetime(group_df['date_local'])



group_df.head()
atl_df = group_df[group_df['county_name'] == 'Fulton']



sns.lineplot(x= 'date', y = 'o3_mean', data = atl_df)
# defining some helpers for testing

min_date = '2010-01-01'

max_date = '2019-12-31'

# counties = "('Davidson','Fulton','Dade','Shelby','Los Angeles')"

# states = "('Tennessee', 'New York', 'California')"



pollutants = ['o3', 'co', 'no2', 'so2', 'pm25_frm', 'pm10', 'temperature']
base_query = """

select 

    state_name

    ,county_name

    ,city_name

    ,cbsa_name as metro_area

    ,latitude

    ,longitude

    ,site_num

    ,date_local

    ,sample_duration

    ,max(aqi) as {}_aqi

    ,max(arithmetic_mean) as {}_mean

from `bigquery-public-data.epa_historical_air_quality`.{}_daily_summary

where date_local >= '{}'

    and date_local <= '{}'

group by state_name

    ,county_name

    ,city_name

    ,cbsa_name 

    ,latitude

    ,longitude

    ,site_num

    ,date_local

    ,sample_duration

order by state_name, county_name, site_num, date_local

"""
# create a dictionary of dataframes containing relevant info

# will write each of these to a flat file, and we can join them to create our daily set of measures for each location

dfs = dict()



for pol in pollutants:

    print('Starting on {}'.format(pol))

    dfs[pol] = bq_assistant.query_to_pandas(base_query.format(pol, pol, pol, min_date, max_date))

    

dfs.keys()
# save these into output

# everything after this is irrelevant

for pol in dfs.keys():

    dfs[pol].to_csv(pol + '.csv', index=False)
# need to pay attention to these sample durations - duplicated measurements for some pollutants

dfs['co']['sample_duration'].value_counts()
# dfs['co'] = dfs['co'][dfs['co']['sample_duration'] == '8-HR RUN AVG END HOUR']
# join_cols = ['metro_area','state_name','county_name','city_name','latitude',

#              'longitude','site_num','date_local', 'poc']



# full_df = dfs['o3']\

#     .merge(dfs['co'], on = join_cols, how = 'outer')\

#     .merge(dfs['no2'], on = join_cols, how = 'outer')
# dfs['no2']['poc'].value_counts()
# getting mean readings for each area - definitely easier to do this in BigQuery/SQL with Pandas inefficiencies

# agg_df = full_df\

#     .groupby(['metro_area','date_local'], as_index=False)\

#     .agg({'latitude':pd.Series.mean,'longitude':pd.Series.mean,'site_num':pd.Series.count,

#          'o3_mean':pd.Series.mean,'o3_aqi':pd.Series.mean,'co_mean':pd.Series.mean,'co_aqi':pd.Series.mean,

#          'no2_mean':pd.Series.mean, 'no2_aqi':pd.Series.mean})
# print('{} total rows'.format(agg_df.shape[0]))

# agg_df[(~agg_df['o3_aqi'].isnull()) & (~agg_df['co_aqi'].isnull()) & (~agg_df['no2_aqi'].isnull())]