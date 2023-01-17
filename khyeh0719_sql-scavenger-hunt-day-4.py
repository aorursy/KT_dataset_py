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
'''
How many Bitcoin transactions were made each day in 2017?
    * You can use the "timestamp" column from the "transactions" table to 
    answer this question. 
    You can check the [notebook from Day 3](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3/) 
    for more information on timestamps.
'''
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp))=2017
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month
            FROM time
            GROUP BY month, day
            ORDER BY month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day_in_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
plt.plot(transactions_per_day_in_2017.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)
'''
How many transactions are associated with each merkle root?
    * You can use the "merkle_root" and "transaction_id" columns in the "transactions" 
    table to answer this question. 
    * Note that the earlier version of this question asked "How many *blocks* are associated 
    with each merkle root?", which would be one block for each root. Apologies for the confusion!
'''
query = """ WITH cte AS 
            (
                SELECT transaction_id,
                    merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT merkle_root AS merkle_root_names,
                COUNT(transaction_id) AS transactions
            FROM cte
            GROUP BY merkle_root
            ORDER BY COUNT(transaction_id) DESC
        """

# note that max_gb_scanned is set to 21, rather than 1
merkle_root_associated = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
print(merkle_root_associated)

plt.plot(merkle_root_associated.transactions)
plt.title("Transcations for each merkle_root")