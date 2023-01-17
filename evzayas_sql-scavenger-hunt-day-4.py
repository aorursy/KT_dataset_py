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
# How many Bitcoin transactions were made each day in 2017?
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            GROUP BY month, day
            ORDER BY month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_daily_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

# plot daily bitcoin transactions
plt.plot(transactions_daily_2017.transactions)
plt.title("2017 Bitcoin Transations by Day")

#  How many transactions are associated with each merkle root?
#  You can use the "merkle_root" and "transaction_id" columns in the "transactions" table. 
query = """ 
            SELECT merkle_root, COUNT(transaction_id) AS count
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
        """

# note that max_gb_scanned is set to 21, rather than 1
merkle_root_trans = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
merkle_root_trans.count()