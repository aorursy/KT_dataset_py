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
bitcoin_blockchain.head("transactions")
query1 = """ WITH time AS 
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
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day 
            ORDER BY year, month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)

transactions_per_day
# plot daily bitcoin transactions in 2017
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
bitcoin_blockchain.head("transactions")
query2 = """  WITH timetable AS
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, 
                     transaction_id, 
                     merkle_root AS MerkleTree
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
            SELECT COUNT(transaction_id) AS NumOfTransactions, MerkleTree,
                EXTRACT(DAY FROM transaction_time) AS Day,
                EXTRACT(MONTH FROM transaction_time) AS Month,
                EXTRACT(YEAR FROM transaction_time) AS Year
            FROM timetable
            GROUP BY MerkleTree, Year, Month, Day
            ORDER BY NumOfTransactions DESC
          """
# Estimate query size
bitcoin_blockchain.estimate_query_size(query2)
# Note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
transactions_per_merkle.head(10)
# plot monthly bitcoin transactions
plt.plot(transactions_per_merkle.NumOfTransactions)
plt.title("Bitcoin Transactions by Merkle Root")