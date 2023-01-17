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

import bq_helper

bitcoin_blockchian = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name = "bitcoin_blockchain")


bitcoin_blockchain.list_tables()
bitcoin_blockchain.head('transactions')
query = """WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            GROUP BY day
            ORDER BY transactions ASC
        """
most_transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=20.8)
most_transactions.head()
import matplotlib.pyplot as plt

plt.plot(most_transactions.transactions)
plt.title("Daily Bitcoin Transcations")
query = """WITH trans_per_block AS
            (
            SELECT COUNT(transaction_id) AS trans_count, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            )
            SELECT trans_count, merkle_root
            FROM trans_per_block
            ORDER BY trans_count DESC
        """
trans_per_block = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=50)
trans_per_block.head(20)
trans_per_block.count()
trans_per_block_over_4500 = trans_per_block["trans_count"] > 4500
trans_per_block[trans_per_block_over_4500].count()
trans_per_block[trans_per_block_over_4500]
import matplotlib.pyplot as plt

plt.plot(trans_per_block[trans_per_block_over_4500].trans_count)
plt.title("transactions associated with merkle root (over 4500 transactions)")

