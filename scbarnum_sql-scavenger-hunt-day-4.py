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
#transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
#transactions_per_month
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)
bitcoin_blockchain.list_tables()
bitcoin_blockchain.table_schema('transactions')

query_1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
            )
            SELECT COUNT(transaction_id) AS number_of_transactions,
                EXTRACT(DAY FROM trans_time) AS day
                FROM time
                WHERE EXTRACT(YEAR FROM trans_time) = 2017
                GROUP BY EXTRACT(DAY FROM trans_time)"""

bitcoin_blockchain.estimate_query_size(query_1)
bitcoin_blockchain.query_to_pandas_safe(query_1)

query_2 = """SELECT COUNT(transaction_id) AS number_of_transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
                GROUP BY merkle_root"""

bitcoin_blockchain.estimate_query_size(query_2)
bitcoin_blockchain.query_to_pandas_safe(query_2)

