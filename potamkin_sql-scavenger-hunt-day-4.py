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
# How many Bitcoin transactions were made each day in 2017?

query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id AS trans_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(trans_id) AS transactions,
            EXTRACT(DAY FROM trans_time) AS day,
            EXTRACT(MONTH FROM trans_time) AS month,
            EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month, day 
            HAVING year = 2017
            ORDER BY year        
        """

# Note that using 'WHERE year = 2017' in place of 'HAVING year = 2017' caused an error.
# Inserted month into the table below because without it, the day count stopped at
# 31. 

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day
# How many transactions are associated with each merkle root?

query = """ SELECT COUNT(transaction_id) as trans_id,
                merkle_root 
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY trans_id DESC
        """

trans_per_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
trans_per_root