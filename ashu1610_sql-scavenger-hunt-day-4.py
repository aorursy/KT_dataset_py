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
query1 = """
         WITH day AS
         ( SELECT TIMESTAMP_MILLIS(timestamp) as trans_time, transaction_id
         FROM `bigquery-public-data.bitcoin_blockchain.transactions`
         WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017     
         )
         SELECT COUNT(transaction_id) AS Transactions,
         EXTRACT(DAYOFYEAR FROM trans_time) AS Transaction_Day
         FROM day
         GROUP BY Transaction_Day
         ORDER BY Transaction_Day
         """

query_1 = bitcoin_blockchain.query_to_pandas(query1)

query_1.head()

import matplotlib.pyplot as plt
plt.plot(query_1.Transaction_Day, query_1.Transactions)
plt.title("Number Of Transactions per day")

query2 = """
         SELECT merkle_root, count(transaction_id) AS Transactions
         FROM `bigquery-public-data.bitcoin_blockchain.transactions`
         GROUP BY merkle_root
         ORDER BY Transactions
         """

query_2 = bitcoin_blockchain.query_to_pandas(query2)

query_2.head()
