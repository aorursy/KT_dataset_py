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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

# How many Bitcoin transactions were made each day in 2017?
bitcoin_blockchain.head("transactions")

query_1 = """WITH time AS (
    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
    )
SELECT COUNT(*) AS transactions, EXTRACT(DAY FROM trans_time) AS day, EXTRACT(MONTH FROM trans_time) AS month
FROM time 
GROUP BY month, day
ORDER BY month, day
"""
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_1, max_gb_scanned=21)
transactions_per_day

# How many transactions are associated with each merkle root?

query_2 = """SELECT COUNT(*) AS transactions, merkle_root 
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY transactions DESC
"""
merkles = bitcoin_blockchain.query_to_pandas_safe(query_2, max_gb_scanned=21)
merkles
