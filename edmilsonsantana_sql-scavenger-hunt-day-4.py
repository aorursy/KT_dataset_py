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
print("How many Bitcoin transactions were made each day in 2017")

query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT
                EXTRACT(YEAR FROM trans_time) as year,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day,
                COUNT(transaction_id) AS transactions
            FROM time
            GROUP BY day, month, year
            HAVING year = 2017
            ORDER BY month, day
        """

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day.head()
print("How many transactions are associated with each merkle root")

query = """WITH merkle_tree AS 
            (
                SELECT merkle_root, transaction_id
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT merkle_root AS root_node, COUNT(transaction_id) AS transactions
            FROM merkle_tree
            GROUP BY root_node
            ORDER BY transactions DESC
        """

transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
transactions_per_merkle_root.head()