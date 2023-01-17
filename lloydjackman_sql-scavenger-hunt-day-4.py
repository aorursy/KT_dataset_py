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
query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month, day
            HAVING year = 2017
            ORDER BY year, month, day
        """
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
transactions_per_day_2017
plt.plot(transactions_per_day_2017.transactions)
plt.title("Daily 2017 Bitcoin Transcations")
transactions_per_day_2017.sample(5)
query3 = """ SELECT merkle_root,
                COUNT(transaction_id) AS transaction_count
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transaction_count DESC
        """
merkle_root_trans_count = bitcoin_blockchain.query_to_pandas(query3)
merkle_root_trans_count.head()