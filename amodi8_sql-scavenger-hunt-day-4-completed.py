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
# Question 1: How many Bitcoin transactions were made each day in 2017?
query1 =""" WITH time AS
            (
            SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, 
                    transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                    trans_time
            FROM time
            GROUP BY trans_time
            ORDER BY trans_time
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas(query1)
# Time-series plot of number of transactions per day
plt.plot(transactions_per_day.trans_time, transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
# Question 2: How many transactions are associated with each merkle root?
query2 =""" SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
        """
transactions_per_merklert = bitcoin_blockchain.query_to_pandas(query2)
# Preview of results from query stored in the transactions_per_merklert dataframe
transactions_per_merklert.head()