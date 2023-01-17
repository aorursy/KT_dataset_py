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
query_transaction= """ WITH transaction_day AS
                       (
                       SELECT TIMESTAMP_MILLIS(timestamp) as time1, transaction_id
                       FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                       )
                       SELECT COUNT(transaction_id) AS transactions,
                       EXTRACT(DAY FROM time1) AS day,
                       EXTRACT(MONTH FROM time1) AS month
                       FROM transaction_day
                       WHERE EXTRACT(YEAR FROM time1) = 2017
                       GROUP BY month, day
                       ORDER BY month, day
                   """ 
transactions_2017=bitcoin_blockchain.query_to_pandas_safe(query_transaction, max_gb_scanned=21)
print(transactions_2017.head())


plt.plot(transactions_2017.transactions)
plt.title("Bitcoin Transactions in 2017")
query_merkle_root= """ SELECT COUNT(transaction_id) AS Number, merkle_root 
                       FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                       GROUP BY merkle_root
                       ORDER BY Number DESC
                   """
transactions_merkle_root=bitcoin_blockchain.query_to_pandas(query_merkle_root)
print(transactions_merkle_root.head())