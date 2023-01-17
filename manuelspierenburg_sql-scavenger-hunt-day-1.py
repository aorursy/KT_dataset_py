# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper  # kaggle big query helper

#BigQuery Table: bigquery-public-data.openaq.global_air_quality
openaq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                       dataset_name="openaq")
openaq.list_tables()
openaq.table_schema('global_air_quality')
openaq.head('global_air_quality')


# countries which dont use ppm as measurement
query = """SELECT DISTINCT country 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' """
openaq.estimate_query_size(query)
countries = openaq.query_to_pandas_safe(query)
print(countries)

# pollutant with value 0
query2 = """SELECT DISTINCT pollutant 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """
openaq.estimate_query_size(query2)
pollutants = openaq.query_to_pandas_safe(query2)
print(pollutants)