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
query_day = """WITH time_day AS
               (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                       transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(transaction_id) AS transactions,
                       EXTRACT(DAY FROM trans_time) AS day
                FROM time_day
                GROUP BY day
                ORDER BY day"""
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_day, max_gb_scanned=21)
plt.plot(transactions_per_day.transactions)
plt.title("Day-wise Bitcoin Transcations")
query_merkle = """
                SELECT merkle_root,
                       COUNT(transaction_id) as transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
                ORDER BY merkle_root
                """
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query_merkle, max_gb_scanned=37)
plt.plot(transactions_per_merkle.transactions)
plt.title("Merkleroot-wise Bitcoin Transcations")
