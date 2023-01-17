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

# QUESTION 1

query2 = """ WITH time AS 
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
            ORDER BY year, month, day
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
plt.xlabel('Day')
plt.ylabel('Transactions Per Day')
# QUESTION 2

query3 = """ WITH merkle AS 
            (
                SELECT merkle_root, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions, merkle_root 
            FROM merkle
            GROUP BY merkle_root
            ORDER BY transactions DESC
        """

# note that max_gb_scanned is set to 21, rather than 1
transaction_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=38)
# plot monthly bitcoin transactions
plt.plot(transaction_per_merkle.transactions)
plt.title("Transcations Per Merkle Root")
plt.xlabel('Merkle Root')
plt.ylabel('Transactions')