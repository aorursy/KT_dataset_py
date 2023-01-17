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
# Question 1: transactions per day in 2017
query_2017 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY month, day
            ORDER BY month, day
            """

# note that max_gb_scanned is set to 21, rather than 1
transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(query_2017, max_gb_scanned=21)
transactions_2017.head()
plt.plot(transactions_2017.transactions)
plt.title("Daily Bitcoin Transcations in 2017");
# Question 2: Transactions per merkle root
query_root = """
            SELECT COUNT(transaction_id) AS transactions, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            """

# note that max_gb_scanned is set to 21, rather than 1
transactions_root = bitcoin_blockchain.query_to_pandas_safe(query_root, max_gb_scanned=37)
transactions_root.head()