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
transactions_per_month.head(20)
daily_query = """WITH time AS 
                (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                        transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(transaction_id) AS transactions,
                    EXTRACT(DAYOFYEAR FROM trans_time) AS day
                FROM time
                WHERE EXTRACT(YEAR FROM trans_time) = 2017
                GROUP BY day 
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(daily_query, max_gb_scanned=21)
transactions_per_day.head(20)
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
# How many transactions are associated with each merkle root?

merkle_query = """
            SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root 
            ORDER BY COUNT(transaction_id) DESC
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_merkle = bitcoin_blockchain.query_to_pandas_safe(merkle_query, max_gb_scanned=37)
transactions_merkle.head()