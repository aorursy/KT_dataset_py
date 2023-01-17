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

# ----------------------

query1 = """WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                COUNT(transaction_id) AS transactions             
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day 
            ORDER BY day
         """

print(bitcoin_blockchain.estimate_query_size(query1))
transactions_each_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
import pandas as pd
pd.set_option('display.max_rows', None)
transactions_each_day
# plot daily bitcoin transactions in 2017
plt.plot(transactions_each_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
query2 = """SELECT merkle_root AS root,
            COUNT(transaction_id) AS transactions             
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY root
            ORDER BY COUNT(transaction_id)
         """

print(bitcoin_blockchain.estimate_query_size(query2))
transactions_each_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)
transactions_each_root.shape
transactions_each_root.head()