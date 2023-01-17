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

# Your code goes here :)

# query to find how many Bitcoin transactions were made each day in 2017

bitcoin_each_day_query = """ WITH dat_to_day as 

                                        (SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,

                                        transaction_id

                                        FROM `bigquery-public-data.bitcoin_blockchain.transactions`)

                            SELECT EXTRACT(DAY FROM trans_time) AS day,

                            EXTRACT(MONTH FROM trans_time) AS month,

                            EXTRACT(YEAR FROM trans_time) AS year,

                            COUNT(transaction_id) as transaction

                            FROM dat_to_day

                            GROUP BY year, month, day

                            HAVING year = 2017

                            ORDER BY year, month, day """



# query to pandas (max_gb_scanned is set to 21, rather than)

bitcoin_each_day = bitcoin_blockchain.query_to_pandas_safe(bitcoin_each_day_query, max_gb_scanned=21)



# see result 

print(bitcoin_each_day)



# plot daily bitcoin transactions

plt.plot(bitcoin_each_day.transaction)

plt.title("Daily Bitcoin Transcations")

# query to show how many transactions are associated with each merkle root?

transaction_each_merkle_root_query = """ WITH merkle AS 

                                                (SELECT merkle_root, transaction_id

                                                FROM `bigquery-public-data.bitcoin_blockchain.transactions`)                                        )

                                        SELECT COUNT(transaction_id) AS transactions, merkle_root

                                        FROM merkle

                                        GROUP BY merkle_root

                                        ORDER BY transactions DESC

                                     """



# query to pandas max_gb_scanned is set to 21, rather than 1

transaction_each_merkle_root = bitcoin_blockchain.query_to_pandas_safe(transaction_each_merkle_root_query, max_gb_scanned=40)



# result

print(trans_per_merkle)


