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

# a method to extract all the months and days AND years
# but this requires post-processing in pandas, as shown
# below
query_transactions_per_day = """WITH time AS (
                                    SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time,
                                        transaction_id
                                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                )
                                SELECT COUNT(transaction_id) AS transactions,
                                        EXTRACT(DAY FROM transaction_time) AS day,
                                        EXTRACT(MONTH FROM transaction_time) AS month,
                                        EXTRACT(YEAR FROM transaction_time) AS year
                                FROM time
                                GROUP BY year, month, day
                                ORDER BY year, month, day"""

bitcoin_blockchain.estimate_query_size(query_transactions_per_day)
# a method where the year column is not in the output because only the months and days
# in 2017 are output.
# note to self (or others who might read this): placing the year extraction
# in the WHERE clause precludes it from existing in the SELECT clause
# workarounds probably include (i.e. to be tested) adding it in post-processing
query_transactions_per_day_2017 = """WITH time AS (
                                        SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time,
                                            transaction_id
                                        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                    )
                                    SELECT COUNT(transaction_id) AS transactions,
                                        EXTRACT(DAY FROM transaction_time) AS day,
                                        EXTRACT(MONTH FROM transaction_time) AS month
                                    FROM time
                                    WHERE EXTRACT(YEAR FROM transaction_time) = 2017
                                    GROUP BY month, day
                                    ORDER BY month, day"""
bitcoin_blockchain.estimate_query_size(query_transactions_per_day_2017)
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_transactions_per_day, max_gb_scanned=21)
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transactions")
# a bit of code to extract only the transactions during 2017
# if the SQL output was for all possible years
transactions_per_day_2017 = transactions_per_day[transactions_per_day['year'] == 2017]
plt.plot(transactions_per_day_2017.transactions)
plt.title("Daily Bitcoin Transactions 2017")
# i chose the transactions to be descending to avoid a plot looking like 
# a bunch of noise. the plot below isn't that much better but it allows you
# to visualize a bit more easily how many people have x number of transactions
query_transactions_per_merkle = """SELECT COUNT(transaction_id) AS transactions,
                                            merkle_root
                                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                    GROUP BY merkle_root
                                    ORDER BY transactions DESC"""
bitcoin_blockchain.estimate_query_size(query_transactions_per_merkle)
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query_transactions_per_merkle,
                                                                    max_gb_scanned=37)
plt.plot(transactions_per_merkle.transactions)
plt.title("Transactions Per Merkle Root")
transactions_per_merkle['transactions'].mean()
transactions_per_merkle['transactions'].median()