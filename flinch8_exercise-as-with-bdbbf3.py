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
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, day 
            ORDER BY year, day
        """

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=23)

#Plotting
import matplotlib.pyplot as plt

plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transactions 2017")
query2 = """ 
            SELECT COUNT(transaction_id) AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root 
            ORDER BY merkle_root
        """
transactions_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)

print(transactions_merkle.head())