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

import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

query_TS_internal = """ WITH time_int AS
                        (
                        SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                        transaction_id
                        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                        WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
                        ) 
                        SELECT EXTRACT(DATE FROM trans_time) AS day,
                               EXTRACT(YEAR FROM trans_time) AS Year,
                               COUNT(transaction_id) AS Transactions
                               FROM time_int
                               GROUP BY day,Year
                               Order BY day 
                        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_TS_internal, max_gb_scanned=21)
transactions_per_day.head()
query_MR_internal = """ WITH time_int2 AS
                        (
                        SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                        transaction_id,
                        merkle_root
                        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                         WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
                        ) 
                        SELECT merkle_root,
                               COUNT(transaction_id) AS Transactions
                               FROM time_int2
                               GROUP BY merkle_root
                        """
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query_MR_internal, max_gb_scanned=40)
transactions_per_merkle