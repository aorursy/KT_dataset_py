# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)
import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="bitcoin_blockchain")

#HUNT 1
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head("transactions")

query_daily_transaction = """ WITH time AS 
        (
        SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
        transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id) AS transactions, 
        EXTRACT(DAY FROM trans_time) AS day, 
        EXTRACT (YEAR FROM trans_time) AS year
        FROM time
        GROUP BY year, day 
        HAVING year = 2017
"""
daily_transaction = bitcoin_blockchain.query_to_pandas_safe(query_daily_transaction, max_gb_scanned=21)
daily_transaction.head()
plt.plot(daily_transaction.transactions)

#HUNT 2 merkle_root

query_merkleroot =""" SELECT COUNT(transaction_id) AS transaction,
                    merkle_root
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    GROUP BY merkle_root
                    ORDER BY transaction DESC
"""
merkle_root_table = bitcoin_blockchain.query_to_pandas_safe(query_merkleroot, max_gb_scanned=40)
bitcoin_blockchain.estimate_query_size(query_merkleroot)
merkle_root_table.head()
