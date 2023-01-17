# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head('blocks')
bitcoin_blockchain.head('transactions')
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

# I didn't use the "WITH .. AS.."
query_2017 = """SELECT EXTRACT(MONTH FROM TIMESTAMP_MILLIS(timestamp)) AS month,
                        EXTRACT(DAY FROM TIMESTAMP_MILLIS(timestamp)) AS day, 
                        COUNT(transaction_id) AS transaction_num
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
                GROUP BY month, day
                ORDER BY month, day
                """
transaction2017 = bitcoin_blockchain.query_to_pandas_safe(query_2017, max_gb_scanned=21)
transaction2017
# note that max_gb_scanned is set to 21, rather than 1
plt.figure(figsize = (20,10))
plt.plot(transaction2017.transaction_num)
plt.title("2017 Daily Bitcoin Transcations")
bitcoin_blockchain.head('transactions')
query_blocks = """SELECT merkle_root,
                        COUNT(block_id) AS block_num
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
                """
merkle_blocks = bitcoin_blockchain.query_to_pandas_safe(query_blocks, max_gb_scanned=37)
merkle_blocks