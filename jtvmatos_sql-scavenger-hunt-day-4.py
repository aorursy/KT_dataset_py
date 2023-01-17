# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

bitcoin_blockchain.list_tables()
bitcoin_blockchain.table_schema('transactions')
bitcoin_blockchain.table_schema('blocks')
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
query1 = """ WITH time AS 
            (
                SELECT EXTRACT(DAY FROM TIMESTAMP_MILLIS(timestamp)) AS day,
                    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS year, EXTRACT(MONTH FROM TIMESTAMP_MILLIS(timestamp)) AS month, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions, day, month
            FROM time
            WHERE year = 2017
            GROUP BY month, day
            ORDER BY month, day
        """

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=31)
print(transactions_per_day)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("2017 - Bitcoin Transcations")
# How many blocks are associated with each merkle root?

query2 = """SELECT merkle_root, count(DISTINCT transaction_id) AS transaction
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY merkle_root
        """

merkle_block = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=51)
print(merkle_block)
#merkle_block.head()