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

import bq_helper

# create a helper object for this dataset
bitcoin_blockchain  = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="bitcoin_blockchain")
#Q1: How many Bitcoin transactions were made each day in 2017?
#You can use the "timestamp" column from the "transactions" table to 
#answer this question. You can check the notebook from Day 3 for more information on timestamps.

# CTE query
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS dayofyear,
                EXTRACT(DATE FROM trans_time) AS date
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY dayofyear, date
            ORDER BY dayofyear, date
        """

transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_2017.transactions)
plt.title("Transactions in 2017")

import bq_helper

# create a helper object for this dataset
bitcoin_blockchain  = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="bitcoin_blockchain")

#Q2: How many transactions are associated with each merkle root?
#You can use the "merkle_root" and "transaction_id" columns in the "transactions" table 
#to answer this question. 

query2 = """ WITH merkle AS 
            (
                SELECT merkle_root AS merkle_r,
                    transaction_id AS t_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(t_id) AS transaction_no, merkle_r
            FROM merkle
            GROUP BY merkle_r
        """

transactions_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)
display(transactions_merkle)