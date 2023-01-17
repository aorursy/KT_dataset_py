# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                  dataset_name='openaq')

# select distinct countries where units are not ppm
query_1 = """SELECT DISTINCT country, unit
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE unit != 'ppm'
          """
countries_without_ppm = open_aq.query_to_pandas_safe(query_1)

# print countries without ppm as unit of measure
countries_without_ppm
# select distict pollutants with value of zero
query_2 = """SELECT DISTINCT pollutant
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE value = 0
          """
zero_value_pollutants = open_aq.query_to_pandas_safe(query_2)

# print pollutants with value of zero
zero_value_pollutants