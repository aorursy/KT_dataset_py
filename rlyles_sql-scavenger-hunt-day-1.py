from google.cloud import bigquery
from bq_helper import BigQueryHelper
import pandas as pd
firstquery = """
SELECT DISTINCT country, unit
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
"""
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(firstquery)
df
secondquery = '''
SELECT DISTINCT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
'''
df2 = bq_assistant.query_to_pandas(secondquery)
df2
