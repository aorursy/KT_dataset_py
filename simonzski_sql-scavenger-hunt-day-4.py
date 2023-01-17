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

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)
query_bitcoin_transactions_day = """WITH time AS
    (
        SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
    SELECT COUNT(transaction_id) AS count, EXTRACT(DAY FROM trans_time) AS day,
        EXTRACT(MONTH FROM trans_time) AS month
    FROM time
    WHERE EXTRACT(YEAR FROM trans_time) = 2017
    GROUP BY month, day
    ORDER BY month, day
    """
bitcoin_transactions_day = bitcoin_blockchain.query_to_pandas_safe(query_bitcoin_transactions_day, max_gb_scanned=21)
bitcoin_transactions_day.tail()
query_blocks_per_merkle_root = """SELECT COUNT(block_id) as block_count, merkle_root
    FROM `bigquery-public-data.bitcoin_blockchain.blocks`
    GROUP BY merkle_root
    ORDER BY block_count DESC"""
blocks_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query_blocks_per_merkle_root)
blocks_per_merkle_root.head()