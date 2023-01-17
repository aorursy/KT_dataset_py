import pandas as pd

from bq_helper import BigQueryHelper
bq_assist = BigQueryHelper('bigquery-public-data', 'bitcoin_blockchain')
bq_assist.table_schema(table_name='transactions')
query = """
#standardSQL
SELECT count(distinct O.output_pubkey_base58) as n_output_addr
FROM `bigquery-public-data.bitcoin_blockchain.transactions` T
CROSS JOIN UNNEST(outputs) O
"""
res = bq_assist.query_to_pandas_safe(query, max_gb_scanned=100)
print("{} addresses received bitcoin at least once.".format(res.n_output_addr[0]))

query = """
#standardSQL
SELECT count(distinct I.input_pubkey_base58) as n_input_addr
FROM `bigquery-public-data.bitcoin_blockchain.transactions` T
CROSS JOIN UNNEST(inputs) I
"""
res = bq_assist.query_to_pandas_safe(query, max_gb_scanned=100)
print("{} addresses sent bitcoin at least once.".format(res.n_input_addr[0]))
query = """
#standardSQL
SELECT input_pubkey_base58, count(*) as count
FROM (
    SELECT I.input_pubkey_base58, T.transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions` T
    CROSS JOIN UNNEST(inputs) I
    ) T
GROUP BY input_pubkey_base58
ORDER BY count(*) DESC
LIMIT 10
"""
res = bq_assist.query_to_pandas_safe(query, max_gb_scanned=100)
print("Most active senders.")
res.head(n=10)
query = """
#standardSQL
SELECT output_pubkey_base58, count(*) as count
FROM (
    SELECT O.output_pubkey_base58, T.transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions` T
    CROSS JOIN UNNEST(outputs) O
    ) T
GROUP BY output_pubkey_base58
ORDER BY count(*) DESC
LIMIT 10
"""
res = bq_assist.query_to_pandas_safe(query, max_gb_scanned=100)
print("Most active receivers.")
res.head(n=10)
query = """
#standardSQL
SELECT output_pubkey_base58, sum(output_satoshis) as count
FROM (
    SELECT O.output_pubkey_base58, T.transaction_id, O.output_satoshis
    FROM `bigquery-public-data.bitcoin_blockchain.transactions` T
    CROSS JOIN UNNEST(outputs) O
    ) T
GROUP BY output_pubkey_base58
ORDER BY sum(output_satoshis) DESC
LIMIT 10
"""
res = bq_assist.query_to_pandas_safe(query, max_gb_scanned=100)
print("Biggest receivers.")
res.head(n=10)