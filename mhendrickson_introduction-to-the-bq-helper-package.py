import pandas as pd

# https://github.com/SohierDane/BigQuery_Helper

from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "openaq")
bq_assistant.list_tables()
bq_assistant.head("global_air_quality", num_rows=3)
bq_assistant.table_schema("global_air_quality")
QUERY = "SELECT location, timestamp, pollutant FROM `bigquery-public-data.openaq.global_air_quality`"
bq_assistant.estimate_query_size(QUERY)
df = bq_assistant.query_to_pandas(QUERY)
df.head(3)
df = bq_assistant.query_to_pandas_safe(QUERY)
df = bq_assistant.query_to_pandas_safe(QUERY, max_gb_scanned=1/10**6)