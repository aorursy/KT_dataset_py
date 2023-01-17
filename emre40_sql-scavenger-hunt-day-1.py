from google.cloud import bigquery
import pandas as pd
from bq_helper import BigQueryHelper

client = bigquery.Client()
QUERY = """
    select distinct country
FROM `bigquery-public-data.openaq.global_air_quality`
where lower(unit) <>'ppm'
"""
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(QUERY)
print(df)
QUERY2 = """
    select pollutant,count(*) as count
FROM `bigquery-public-data.openaq.global_air_quality`
where value=0
group by pollutant
"""
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(QUERY2)
print(df)