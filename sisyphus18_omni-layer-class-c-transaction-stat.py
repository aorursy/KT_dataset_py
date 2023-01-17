import pandas as pd
from bq_helper import BigQueryHelper

# This establishes an authenticated session and prepares a reference to the dataset that lives in BigQuery.
bq_assistant = BigQueryHelper("bigquery-public-data", "ethereum_blockchain")
# class c usdt send to owners tx, all invalid
query = """
SELECT
  transaction.transaction_id
FROM
  `bigquery-public-data.bitcoin_blockchain.transactions` AS transaction
JOIN
  UNNEST (outputs) AS outputs
where
 STARTS_WITH(outputs.output_script_string, "RETURN PUSHDATA(20)[6f6d6e69000000030000001f")

"""
df = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=100)
df
# class c usdt simple send tx
query = """
SELECT
  count(1)
FROM
  `bigquery-public-data.bitcoin_blockchain.transactions` AS transaction
JOIN
  UNNEST (outputs) AS outputs
where
 STARTS_WITH(outputs.output_script_string, "RETURN PUSHDATA(20)[6f6d6e69000000000000001f")

"""
df = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=100)
df
# class c send all tx
query = """
SELECT
  transaction.transaction_id
FROM
  `bigquery-public-data.bitcoin_blockchain.transactions` AS transaction
JOIN
  UNNEST (outputs) AS outputs
where
 STARTS_WITH(outputs.output_script_string, "RETURN PUSHDATA(9)[6f6d6e690000000401")
limit 10

"""
df = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=100)
df
print(df.values)