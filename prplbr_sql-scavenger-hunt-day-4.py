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
transactions_per_month.head()
# import plotting library
import matplotlib.pyplot as plt
import datetime
import pandas as pd
date=[datetime.datetime(y,m,1) for y,m in zip(transactions_per_month.year,transactions_per_month.month)]

# plot monthly bitcoin transactions
plt.plot(date,transactions_per_month.transactions,'.')
plt.title("Monthly Bitcoin Transactions")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.table_schema('transactions')
# Your code goes here :)
query1= """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS n_transactions,
                EXTRACT(YEAR FROM trans_time) AS year,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year,month,day 
            ORDER BY month,day

        """
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
transactions_per_day_2017.head()
date=[datetime.datetime(y,m,d) for y,m,d in zip(transactions_per_day_2017.year,transactions_per_day_2017.month,transactions_per_day_2017.day)]
plt.plot(date,transactions_per_day_2017.n_transactions,'.')
query2= """ SELECT COUNT(transaction_id) AS n_transactions,
            merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root 
            ORDER BY n_transactions DESC
        """
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
transactions_per_merkle.head()
transactions_per_merkle.head(30).plot(kind='bar')