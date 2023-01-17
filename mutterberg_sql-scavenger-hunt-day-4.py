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
plt.title("Monthly Bitcoin Transactions")
query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month, day
            ORDER BY year, month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
transactions_per_day.to_csv("daily_transactions.csv")
# Dataset is from 2009, not 2017?

# plot 2017 daily bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transactions")
query3 = """ WITH time AS 
            (
                SELECT merkle_root,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions
            FROM time
            GROUP BY merkle_root
        """

# note that max_gb_scanned is set to 40, rather than 21 or 1 (estimated size of 36.94322868436575)
merkle_count = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=40)
merkle_count.to_csv("merkle_count.csv")