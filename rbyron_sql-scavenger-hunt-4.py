# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head('transactions')
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time)=2017
            GROUP BY day 
            ORDER BY day
        """

# max_gb_scanned is set to 21
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=25)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
query = """ SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
        """

# max_gb_scanned is set to 21
transactions_per_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
transactions_per_root.shape
transactions_per_root.head()
transactions_per_root['transactions'].min()
transactions_per_root['transactions'].max()