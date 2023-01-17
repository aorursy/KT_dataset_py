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
            SELECT FORMAT_TIMESTAMP('%j', trans_time) AS day, 
                   COUNT(transaction_id) AS transactions
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day 
            ORDER BY day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

transactions_per_day
# import plotting library
import matplotlib.pyplot as plt

# plot daily bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
query = """SELECT merkle_root AS root, 
                   COUNT(block_id) AS blocks
            FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            GROUP BY root
            ORDER BY blocks DESC
        """

# note that max_gb_scanned is set to 21, rather than 1
blocks_per_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

blocks_per_root