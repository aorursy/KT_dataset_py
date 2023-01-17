from google.cloud import bigquery
import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper
# Use  bq_helper to create a BigQueryHelper object
bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
%%time
bq_assistant.table_schema("languages")
%%time
bq_assistant.head("languages", num_rows=20)
