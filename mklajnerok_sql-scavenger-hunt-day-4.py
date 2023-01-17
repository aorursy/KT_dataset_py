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
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head('transactions')
bitcoin_blockchain.table_schema('transactions')
# question 1
my_query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day
            ORDER BY day
        """
# estimate before running
bitcoin_blockchain.estimate_query_size(my_query1)
transactions_per_2017_day = bitcoin_blockchain.query_to_pandas_safe(my_query1, max_gb_scanned=21)
transactions_per_2017_day.head()
# different version
my_query1A = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day
            FROM time
            GROUP BY day 
        """

# estimate before running
bitcoin_blockchain.estimate_query_size(my_query1A)
# different version
my_query1B = """ 
                SELECT COUNT(transaction_id) AS transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
                GROUP BY EXTRACT(DAYOFYEAR FROM TIMESTAMP_MILLIS(timestamp))
        """

# estimate before running
bitcoin_blockchain.estimate_query_size(my_query1B)
# question 2
my_query2 = """SELECT COUNT(transaction_id) AS transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP by merkle_root
                ORDER BY transactions DESC"""

# estimate before running
bitcoin_blockchain.estimate_query_size(my_query2)
transactions_per_root = bitcoin_blockchain.query_to_pandas_safe(my_query2, max_gb_scanned=37)
transactions_per_root.head()