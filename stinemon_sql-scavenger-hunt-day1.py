# setup
import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# explore tables
open_aq.list_tables()
open_aq.head('global_air_quality')
query1 = '''SELECT DISTINCT(country)
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        '''
result1 = open_aq.query_to_pandas_safe(query1)
result1
query2 = '''SELECT DISTINCT(pollutant)
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        '''
result2 = open_aq.query_to_pandas_safe(query2)
result2
