import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper(active_project='bigquery-public-data',dataset_name='openaq')

bq_assistant.list_tables()
bq_assistant.head('global_air_quality',num_rows=10)
query = """SELECT *
        From `bigquery-public-data.openaq.global_air_quality`"""
bq_assistant.estimate_query_size(query)
query = """SELECT city 
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country='US'"""
US_cities = bq_assistant.query_to_pandas(query)
bq_assistant.table_schema('global_air_quality')
query = """SELECT country 
        From `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant!='ppm'
        """
countries_not_ppm = bq_assistant.query_to_pandas(query)
countries_not_ppm.head()
query = """SELECT pollutant 
        From `bigquery-public-data.openaq.global_air_quality`
        WHERE value=0.0
        """
pollutant_is_zero = bq_assistant.query_to_pandas(query)
pollutant_is_zero.head(15)
poll