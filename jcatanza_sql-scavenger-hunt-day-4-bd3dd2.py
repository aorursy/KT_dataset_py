# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()

bitcoin_blockchain.table_schema("blocks")

bitcoin_blockchain.table_schema("transactions")
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
# Query number of daily bitcoin transactions for 2017, ordered by day 
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month, day 
            HAVING year = 2017
            ORDER BY year, month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# plot daily bitcoin transactions in 2017
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
# Query for number of transactions for each merkle root
query = """ WITH data AS 
            (
                SELECT merkle_root, block_id,
                     EXTRACT(YEAR FROM  TIMESTAMP_MILLIS(timestamp))
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(block_id) AS transactions, merkle_root   
            FROM data
            GROUP BY merkle_root
            ORDER BY transactions DESC
        """
            
# note that max_gb_scanned is set to 40, rather than 1
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
# plot log of number of transactions per merkle root in descending order
import numpy as np
plt.plot(np.log10(transactions_per_merkle_root.transactions))
plt.title("log 10 of merkle_root bitcoin transactions")