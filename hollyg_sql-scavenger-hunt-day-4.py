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
query_num_trans_2017 = """
WITH time AS (
    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT COUNT(transaction_id) AS transactions,
    EXTRACT(DAYOFYEAR FROM trans_time) AS day_of_year,
    EXTRACT(YEAR FROM trans_time) AS year
    FROM time
    WHERE EXTRACT(YEAR FROM trans_time) = 2017
    GROUP BY year, day_of_year
    ORDER BY year, day_of_year
"""
bitcoin_blockchain.estimate_query_size(query_num_trans_2017)
num_trans_2017 = bitcoin_blockchain.query_to_pandas_safe(query_num_trans_2017, max_gb_scanned=21)
num_trans_2017.head()
# plot monthly bitcoin transactions
plt.plot(num_trans_2017.transactions)
plt.title("Daily Bitcoin Transactions in 2017")
query_num_trans_merkle_root = """
    SELECT count(transaction_id) AS tx_count,
     merkle_root
     FROM `bigquery-public-data.bitcoin_blockchain.transactions`
     GROUP BY merkle_root
     ORDER BY tx_count
"""
bitcoin_blockchain.estimate_query_size(query_num_trans_merkle_root)
num_trans_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query_num_trans_merkle_root, max_gb_scanned=37)
num_trans_merkle_root.head()
plt.plot(num_trans_merkle_root.tx_count)
plt.title("Transactions by Merkle Root")