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
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            GROUP BY month, day
            ORDER BY month, day
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)

plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
query2 =  """ SELECT COUNT(transaction_id) AS transactions,
                merkle_root AS Merkle
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY Merkle
            ORDER BY transactions  DESC
        """
blocks_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
blocks_per_merkle.head()

plt.plot(blocks_per_merkle.transactions)
plt.title("Transactions per Merkle Root")