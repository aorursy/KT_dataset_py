import bq_helper
bitcoinDS = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='bitcoin_blockchain')
bitcoinDS.table_schema('transactions')
bitcoinDS.head('transactions')
query = """
            WITH bitcointrans2017 AS
            (
                SELECT transaction_id, 
                TIMESTAMP_MILLIS(timestamp) AS trans_time
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT EXTRACT(MONTH FROM trans_time) AS month, EXTRACT(DAY FROM trans_time) AS day, COUNT(transaction_id) as transCount
            FROM bitcointrans2017
            GROUP BY month, day
            ORDER BY month, day
"""
bitcoinDS.estimate_query_size(query)
transactions_per_dat_2017 = bitcoinDS.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_dat_2017.head()
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_dat_2017.transCount)
plt.title("daily Bitcoin Transcations")