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
print(transactions_per_month)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# How many Bitcount transactions were made each day in 2017?
query2 = """ WITH time AS 
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
            ORDER BY year, day
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
print(transactions_per_day)
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
# How many transactions are associated with each merkle root?
query3 = """SELECT COUNT(DISTINCT transaction_id) AS transactions, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions
        """
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=37)
print(transactions_per_merkle)
