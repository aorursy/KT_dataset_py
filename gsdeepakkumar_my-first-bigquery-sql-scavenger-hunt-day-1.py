# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper #For processing big queries

openAQ=bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")
openAQ.list_tables()
openAQ.table_schema('global_air_quality')
openAQ.head('global_air_quality')
query = """ SELECT country,unit FROM `bigquery-public-data.openaq.global_air_quality` where unit !='ppm' order by country"""
#Check for query size 
openAQ.estimate_query_size(query)
#Running the query 
country=openAQ.query_to_pandas_safe(query)
# Which country has most number of units != ppm 
country.country.value_counts()
query = """ SELECT pollutant,value FROM `bigquery-public-data.openaq.global_air_quality` where value=0 order by pollutant"""
openAQ.estimate_query_size(query)
pollutants=openAQ.query_to_pandas_safe(query)
# Output first 5 rows
pollutants.head()
pollutants.pollutant.value_counts()