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
# How many Bitcoin transactions were made each day in 2017?
query = """  select date, count_transactions from(
SELECT date(TIMESTAMP_MILLIS(timestamp)) AS date, count(transaction_id) as count_transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`   
                where extract (year from  TIMESTAMP_MILLIS(timestamp)) = 2017
                group by date(TIMESTAMP_MILLIS(timestamp) ))
                order by date
         """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day
# How many transactions are associated with each merkle root
query2 = """  
SELECT merkle_root, count(block_id) as count_transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`   
                group by merkle_root
         """
transactions_per_merkleroot = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
transactions_per_merkleroot