# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = '''with time as
           (
           select timestamp_millis(timestamp) as trans_time, transaction_id
           from `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           select count(transaction_id) as transactions, 
           extract(dayofyear from trans_time) as dayofyear
           from time
           group by dayofyear
           order by transactions
           '''
bitcoin_blockchain.estimate_query_size(query)
trans_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=24)
trans_per_day.head(10)
query = '''select merkle_root as merkle_root, count(transaction_id) as trans_count
           from `bigquery-public-data.bitcoin_blockchain.transactions`
           group by merkle_root
           order by trans_count desc'''
bitcoin_blockchain.estimate_query_size(query)
trans_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=42)
trans_per_merkle.head(10)