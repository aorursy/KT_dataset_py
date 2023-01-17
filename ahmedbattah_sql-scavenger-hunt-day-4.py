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
# Transactions per day in 2017
query_by_day = """
                WITH day2017 AS (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id 
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(transaction_id) AS transactions,
                EXTRACT (DAYOFYEAR FROM trans_time) AS day
                FROM day2017
                WHERE EXTRACT (YEAR FROM trans_time) = 2017
                GROUP BY day
                ORDER BY day
                """
# Query size
bitcoin_blockchain.estimate_query_size(query_by_day)

# Dataframe
trans_by_day = bitcoin_blockchain.query_to_pandas_safe(query_by_day, max_gb_scanned = 21)

# plot 2017 daily bitcoin transactions
plt.plot(trans_by_day.transactions)
plt.title("2017 Daily Bitcoin Transcations")

# Number of transaction by merkle root
merkle_query = """
                WITH merkle AS(
                    SELECT block_id, merkle_root
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(block_id) AS block_id, merkle_root
                FROM merkle
                GROUP BY merkle_root
                ORDER BY block_id
                """

# Query size
bitcoin_blockchain.estimate_query_size(merkle_query)

# Dataframe
merkle_root = bitcoin_blockchain.query_to_pandas_safe(merkle_query, max_gb_scanned = 37)
merkle_root.head()

# plot transactions by merkle root
plt.plot(merkle_root.block_id)
plt.title("Transcations by merkle root")
