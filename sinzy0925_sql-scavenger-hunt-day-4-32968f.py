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
# How many Bitcoin transactions were made each day in 2017?
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            where EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, day
            ORDER BY year, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Dayly Bitcoin Transcations")
#  How many transactions are associated with each merkle root?
# * You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question. 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ WITH merkle_roots AS 
            (
                SELECT merkle_root,transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM merkle_roots
            GROUP BY merkle_root
            --ORDER BY year, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_merkle_root.transactions)
plt.title("each merkle_root Bitcoin Transcations")
