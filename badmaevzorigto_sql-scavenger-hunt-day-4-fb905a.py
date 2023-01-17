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

transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=30)
# import plotting library

import matplotlib.pyplot as plt



# plot monthly bitcoin transactions

plt.plot(transactions_per_month.transactions)

plt.title("Monthly Bitcoin Transcations")
bitcoin_blockchain.head('transactions')
# Задание 1

query = """ WITH time AS 

            (

                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,

                    transaction_id

                FROM `bigquery-public-data.bitcoin_blockchain.transactions`

            )

            SELECT COUNT(transaction_id) as transactions

            FROM time

            WHERE EXTRACT(YEAR FROM trans_time)=2017

            GROUP BY EXTRACT(DAYOFYEAR FROM trans_time)

        """

transactions_per_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=30)

plt.plot(transactions_per_2017.transactions)
# Задание 2

query2 = """SELECT COUNT(transaction_id) as transactions, merkle_root

            FROM `bigquery-public-data.bitcoin_blockchain.transactions`

            GROUP BY merkle_root

        """

data = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=45)

data.head(10)