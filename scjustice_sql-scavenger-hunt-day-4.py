# import package with helper functions 

import bq_helper



# create a helper object for this dataset

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                              dataset_name="crypto_bitcoin")
query = """ WITH time AS 

            (

                SELECT block_timestamp, "hash" AS transaction_id

                FROM `bigquery-public-data.crypto_bitcoin.transactions`

            )

            SELECT COUNT(transaction_id) AS transactions,

                EXTRACT(MONTH FROM block_timestamp) AS month,

                EXTRACT(YEAR FROM block_timestamp) AS year

            FROM time

            GROUP BY year, month 

            ORDER BY year, month

        """



transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=5)
# import plotting library

import matplotlib.pyplot as plt



# plot monthly bitcoin transactions

plt.plot(transactions_per_month.transactions)

plt.title("Monthly Bitcoin Transcations")
# Determine the number of transactions each day of 2017



query = """ WITH time AS

            (

                SELECT EXTRACT(dayofyear FROM block_timestamp) AS day,

                    EXTRACT(year FROM block_timestamp) AS year,

                    'hash' AS transaction_id

                FROM `bigquery-public-data.crypto_bitcoin.transactions`

            )

            SELECT COUNT(transaction_id) AS transactions, 

                day, year

            FROM time 

            WHERE year = 2017

            GROUP BY year, day

            ORDER BY year, day

        """



bitcoin_blockchain.estimate_query_size(query)
transactions_by_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=5)
import matplotlib.pyplot as plt



plt.plot(transactions_by_day.transactions)

plt.title('Bitcoin Transaction Count by Day During 2017')
query = """ WITH blocks AS

            (

                SELECT merkle_root, transaction_count

                FROM `bigquery-public-data.crypto_bitcoin.blocks`

            )

            SELECT merkle_root, SUM(transaction_count) AS num_transactions

                FROM blocks

                GROUP BY merkle_root

                ORDER BY num_transactions DESC

        """



print(bitcoin_blockchain.estimate_query_size(query))

merkle_transactions = bitcoin_blockchain.query_to_pandas_safe(query)
merkle_transactions.head()