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
transactions_per_month.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, day
            HAVING year = 2017
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
# How many transactions are associated with each merkle root?
query2 ="""
            SELECT COUNT(transaction_id) AS transactions, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions DESC
            LIMIT 20
        """
merkle_root_transactions = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
merkle_root_transactions.head(10)
query2 = """WITH merkle AS
            (
            SELECT COUNT(transaction_id) AS transactions,
                    merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            )
            SELECT merkle_root, transactions
            FROM merkle
        """
trans_per_merkleroot = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)
trans_per_merkleroot.head(10)