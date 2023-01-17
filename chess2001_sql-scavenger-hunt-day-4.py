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

#check https://www.epochconverter.com

#Epoch timestamp: 1483228800
#Timestamp in milliseconds: 1483228800000
#Human time (GMT): Sunday, January 1, 2017 12:00:00 AM
#begin = 1483228800000 - 1

#Timestamp in milliseconds: 1514768400000
#GMT: Monday, January 1, 2018 1:00:00 AM
#Your time zone: Monday, January 1, 2018 3:00:00 AM GMT+02:00
#end = 1514768400000


query_per_day_2017 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE timestamp > 1483228800000 - 1 AND timestamp < 1514768400000
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            GROUP BY month, day 
            ORDER BY month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query_per_day_2017, max_gb_scanned=21)


transactions_per_day_2017

#You can use the "merkle_root" and "transaction_id" columns in the "transactions" table 
#to answer this question.
#Note that the earlier version of this question asked 
#"How many blocks are associated with each merkle root?", which would be one block for each root.

query_merkle = """
        SELECT merkle_root, count(transaction_id) as transactions_cnt
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        ORDER BY transactions_cnt DESC
        """
df_merkle = bitcoin_blockchain.query_to_pandas_safe(query_merkle, max_gb_scanned = 40)
df_merkle.head(15)