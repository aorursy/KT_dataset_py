from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

query = """
#standardSQL
SELECT
  o.year,
  COUNT(DISTINCT(o.output_key)) AS recipients
FROM (
  SELECT
    TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          365*86400000))) AS year,
    output.output_pubkey_base58 AS output_key
  FROM
    `bigquery-public-data.bitcoin_blockchain.transactions`,
    UNNEST(outputs) AS output ) AS o
GROUP BY
  year
ORDER BY
  year
  """

query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
transactions.head(10)
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline

transactions.plot()
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper

# This establishes an authenticated session and prepares a reference to the dataset that lives in BigQuery.
bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")
df = bq_assistant.query_to_pandas_safe(query)
df = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=30)
print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))
df.plot()