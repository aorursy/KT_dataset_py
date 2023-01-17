# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """WITH trans_time AS
           (
               SELECT transaction_id AS id, TIMESTAMP_MILLIS(timestamp) AS time
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT EXTRACT(YEAR FROM time) AS year, COUNT(id) AS num_trans
           FROM trans_time
           WHERE EXTRACT(YEAR FROM time) = 2017
           GROUP BY EXTRACT(YEAR FROM time)"""
df = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)
print(df)
query = """SELECT merkle_root, COUNT(transaction_id) AS num_trans
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY merkle_root
           ORDER BY num_trans DESC"""
df = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=39)
print(df)
