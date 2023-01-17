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
query = """ WITH days AS
                (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                        EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS YEAR,
                        transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
            SELECT EXTRACT(DAYOFYEAR FROM trans_time) AS day_year,
                COUNT(transaction_id)
            FROM days
            WHERE YEAR = 2017
            GROUP BY day_year
            ORDER BY day_year
            """
bitcoin_blockchain.estimate_query_size(query)
df_year = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
df_year
query = """ SELECT merkle_root, COUNT(transaction_id) AS trans
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY trans DESC
        """
bitcoin_blockchain.estimate_query_size(query)
df_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
df_merkle