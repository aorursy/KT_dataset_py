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
# How many Bitcoin transactions were made each day in 2017?
query_day = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT 
                EXTRACT(month FROM trans_time) AS month,
                EXTRACT(day FROM trans_time) AS day,
                COUNT(transaction_id) AS transactions
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day, month 
            ORDER BY month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_day, max_gb_scanned=21)
# plot monthly bitcoin transactions

print(transactions_per_day.head(13))
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transactions - 2017")
# How many transactions are associated with each merkle root?
query_merkle = """ 
            SELECT merkle_root, COUNT(transaction_id) AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root 
            ORDER BY  COUNT(transaction_id) Desc
        """

# note that max_gb_scanned is set to 21, rather than 1
merkles = bitcoin_blockchain.query_to_pandas_safe(query_merkle, max_gb_scanned=37)
# plot monthly bitcoin transactions
print(merkles.head(20))

print(merkles)