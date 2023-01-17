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
# Questions 1: How many Bitcoin transactions were made each day in 2017?
query1 = """WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT EXTRACT(YEAR FROM trans_time) AS year,
                   EXTRACT(MONTH FROM trans_time) AS month,
                   EXTRACT(DAY FROM trans_time) AS day,
                   COUNT(transaction_id) AS transactions
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day
            ORDER By year, month, day                              
        """

transactions_in_2017 = bitcoin_blockchain.query_to_pandas_safe(query=query1,max_gb_scanned=21)
# plot the data
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_in_2017.transactions)
plt.title('Daily Number of Transcations in 2017')
# Question 2: How many transactions are associated with each merkle root?

query2 = """SELECT 
                merkle_root, 
                COUNT(block_id) as blocks
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY merkle_root
        """
blocks_in_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query=query2,max_gb_scanned=40)