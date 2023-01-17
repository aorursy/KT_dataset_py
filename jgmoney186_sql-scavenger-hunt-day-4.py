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
query1 = """  WITH time AS (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time)  = 2017
            GROUP BY month, day 
            ORDER BY month, day
"""
transactions_per_day = bitcoin_blockchain.query_to_pandas(query1)

print(transactions_per_day)
transactions_per_day.to_csv("transactions_per_day.csv")

#use transaction id and merkle root for this one
query2 = """ SELECT merkle_root,
                    COUNT (DISTINCT transaction_id) AS transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
            
            """ 

transactions_per_merkles = bitcoin_blockchain.query_to_pandas(query2) 

print(transactions_per_merkles)
transactions_per_merkles.to_csv("transactions_per_merkles.csv")
