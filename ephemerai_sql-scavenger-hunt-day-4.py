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
from bq_helper import BigQueryHelper
bitcoin = BigQueryHelper(active_project = "bigquery-public-data",
                        dataset_name = "bitcoin_blockchain")
bitcoin.head("transactions", num_rows = 1)
# How many Bitcoin transactions were made each day in 2017?
query1 = """WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, 
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions, 
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day, month
            ORDER BY month, day
         """

# Estimate the size of this query
bitcoin.estimate_query_size(query1)
# Another Way!
query1 = """WITH time AS
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
            GROUP BY day, month, year
            HAVING year = 2017
            ORDER BY month, day
         """

# Estimate the size of this query
bitcoin.estimate_query_size(query1)
transaction_daily_2017 = bitcoin.query_to_pandas_safe(query1, max_gb_scanned = 21)
transaction_daily_2017.head(50)
# import library
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

# plot
plt.plot(transaction_daily_2017.transactions)
plt.title("Daily Bitcoin Transactions in 2017")
# How many transactions are associated with each merkle root?
query2 = """
            SELECT merkle_root, 
                COUNT(transaction_id) AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions
        """

# Estimate the size of this query
bitcoin.estimate_query_size(query2)
transaction_merkle_root = bitcoin.query_to_pandas_safe(query2, max_gb_scanned = 38)
# Result
transaction_merkle_root