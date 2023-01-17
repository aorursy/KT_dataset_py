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
query1 = """ WITH time AS 
            (
                SELECT transaction_id, TIMESTAMP_MILLIS(timestamp) AS trans_time
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
SELECT EXTRACT(DATE FROM trans_time) AS date, 
       COUNT(transaction_id) AS num_transactions
FROM time
GROUP BY date
ORDER BY date
"""
bitcoin_blockchain.estimate_query_size(query1)
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, 21)
print(transactions_per_day)
plt.plot(transactions_per_day.num_transactions)
plt.title("Daily Bitcoin Transactions")
query2 = """
SELECT merkle_root, COUNT(block_id) AS num_blocks
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY num_blocks
"""
blocks_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, 37)
print(blocks_per_merkle_root)
query2b = """ WITH count_blocks AS (
    SELECT merkle_root, COUNT(block_id) AS num_blocks
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle_root)
SELECT num_blocks, COUNT(merkle_root) AS num_merkles
FROM count_blocks
GROUP BY num_blocks
ORDER BY num_blocks
"""
blocks_per_merkle_distribution = bitcoin_blockchain.query_to_pandas_safe(query2b, 37)
print(blocks_per_merkle_distribution)
from scipy import stats
stats.describe(blocks_per_merkle_distribution.num_merkles)