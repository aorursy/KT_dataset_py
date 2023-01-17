# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
df=bitcoin_blockchain.query_to_pandas(""" 
  WITH time AS (
    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
      FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
    SELECT COUNT(transaction_id) AS transactions,
      EXTRACT(DAYOFYEAR FROM trans_time) AS day
   FROM time
   WHERE EXTRACT(YEAR FROM trans_time)=2017
   GROUP BY day
   ORDER BY day
""")
df
df=bitcoin_blockchain.query_to_pandas("""
  SELECT COUNT(transaction_id) AS transaction_count_by_merkle_root, merkle_root
  FROM `bigquery-public-data.bitcoin_blockchain.transactions`
  GROUP BY merkle_root
  """)
df