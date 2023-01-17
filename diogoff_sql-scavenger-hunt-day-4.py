import bq_helper
blockchain = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                      dataset_name='bitcoin_blockchain')
blockchain.list_tables()
blockchain.head('transactions')
query = '''WITH time AS
           (
               SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT COUNT(transaction_id) AS num_transactions,
                  EXTRACT(MONTH FROM trans_time) AS month,
                  EXTRACT(YEAR FROM trans_time) AS year
           FROM time
           GROUP BY year, month
           ORDER BY year, month'''
blockchain.estimate_query_size(query)
num_transactions_by_month = blockchain.query_to_pandas_safe(query)
num_transactions_by_month = blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
num_transactions_by_month.head()
num_transactions_by_month.plot(x='month', y='num_transactions', kind='bar')
query = '''WITH time AS
           (
               SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT COUNT(transaction_id) AS num_transactions,
                  EXTRACT(DAY FROM trans_time) AS day
           FROM time
           WHERE EXTRACT(YEAR FROM trans_time) = 2017
           GROUP BY day
           ORDER BY day'''
blockchain.estimate_query_size(query)
num_transactions_by_day = blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
num_transactions_by_day.head()
num_transactions_by_day.plot(x='day', y='num_transactions', kind='bar')
blockchain.head('blocks')
query = '''SELECT merkle_root, COUNT(block_id) AS num_blocks
           FROM `bigquery-public-data.bitcoin_blockchain.blocks`
           GROUP BY merkle_root
           ORDER BY num_blocks DESC'''
blockchain.estimate_query_size(query)
num_blocks_by_merkle_root = blockchain.query_to_pandas_safe(query)
num_blocks_by_merkle_root.head()