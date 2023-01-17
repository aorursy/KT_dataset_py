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
# query for question 1
query_1 = """
                WITH transaction_time AS
                (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS transform_time,
                        transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(transaction_id) AS transactions,
                        EXTRACT(DAY FROM transform_time) AS day,
                        EXTRACT(YEAR FROM transform_time) AS year
                FROM transaction_time
                GROUP BY year, day
                HAVING year = 2017
                ORDER BY year, day
"""
bitcoin_blockchain.estimate_query_size(query_1)
query_1_result = bitcoin_blockchain.query_to_pandas_safe(query_1, max_gb_scanned=21)
query_1_result
query_1_result[query_1_result.year == 2018].count()
# import plotting library
import matplotlib.pyplot as plt

# plot daily in mon bitcoin transactions
plt.plot(query_1_result.transactions)
plt.title("Daily Bitcoin Transcations In 2017")
# query for question 2
query_2 = """
                WITH sum_of_transactions AS
                (
                    SELECT COUNT(transaction_id) AS cnt_transactions,
                            merkle_root
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    WHERE merkle_root IS NOT NULL
                    GROUP BY merkle_root
                    ORDER BY cnt_transactions DESC
                )
                SELECT AVG(cnt_transactions) AS avg_transaction
                FROM sum_of_transactions
"""

bitcoin_blockchain.estimate_query_size(query_2)
query_2_result = bitcoin_blockchain.query_to_pandas_safe(query_2, max_gb_scanned=37)
query_2_result.head()