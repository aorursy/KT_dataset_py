import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
%%time
bq_assistant.list_tables()
QUERY = """
        SELECT COUNT(1)
        FROM `bigquery-public-data.github_repos.commits`
        """
%%time
df = bq_assistant.query_to_pandas_safe(QUERY)
print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))
df.head()
QUERY = """
        SELECT license, COUNT(1)
        FROM `bigquery-public-data.github_repos.licenses`
        GROUP BY license
        ORDER BY 2 DESC
        """
%%time
df = bq_assistant.query_to_pandas_safe(QUERY)
df.head()
# print information on all the columns in the "commits" table
# in the github repos dataset
bq_assistant.table_schema("commits")
#standardSQL
QUERY = """
SELECT
   encoding,COUNT(1)
FROM
  `bigquery-public-data.github_repos.commits`
GROUP BY encoding
ORDER BY 2 DESC
"""
%%time
df = bq_assistant.query_to_pandas_safe(QUERY)
df.head()
