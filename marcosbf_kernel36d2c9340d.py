import pandas as pd

# https://github.com/SohierDane/BigQuery_Helper

from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "openaq")
bq_assistant.list_tables()
bq_assistant.head("global_air_quality", num_rows=3)