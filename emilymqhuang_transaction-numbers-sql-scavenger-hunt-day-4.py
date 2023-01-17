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
# My code goes here :)
# 1. How many Bitcoin transactions were made each day in 2017?
query_1 = """WITH trans_time AS
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS time, transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
             SELECT EXTRACT(YEAR FROM time) AS year
                    , EXTRACT(MONTH FROM time) AS month 
                    , EXTRACT(DAY FROM time) AS day
                    , COUNT(transaction_id) AS number_of_transactions
             FROM trans_time
             WHERE EXTRACT(YEAR FROM time) = 2017
             GROUP BY year, month, day
             ORDER BY year, month, day
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_1, max_gb_scanned=21)
transactions_per_day.head()
# plot daily bitcoin transactions for Year 2017
plt.plot(transactions_per_day.number_of_transactions)
plt.title("2017 Daily Bitcoin Transcations")
# 2.How many transactions are associated with each merkle root?
query_2 = """SELECT merkle_root, COUNT(transaction_id) AS number_of_transactions
             FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             GROUP BY merkle_root
             ORDER BY number_of_transactions DESC
        """
merkle_root_transactions = bitcoin_blockchain.query_to_pandas_safe(query_2, max_gb_scanned=40)
merkle_root_transactions.head()
plt.plot(merkle_root_transactions.number_of_transactions)
plt.title("Number of Transcations Associated with Each Merkle Root")