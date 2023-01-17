# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head("transactions")
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
#Transaction per day in 2017
myquery = """ WITH day AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_date,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
             )
            SELECT COUNT(transaction_id) AS bittransactions,
                EXTRACT(DATE FROM trans_date) AS date
            FROM day
            WHERE EXTRACT (YEAR FROM trans_date) = 2017
            GROUP BY date 
            ORDER BY date
        """



transactions_perday = bitcoin_blockchain.query_to_pandas_safe(myquery, max_gb_scanned=21)
transactions_perday
myquery2 = """ SELECT merkle_root, COUNT (transaction_id) As bittrans
                FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
            GROUP BY merkle_root 
            ORDER BY bittrans DESC
        """
merkletrans = bitcoin_blockchain.query_to_pandas_safe(myquery2, max_gb_scanned = 37)
merkletrans