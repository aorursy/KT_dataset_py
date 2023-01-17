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
# Your code goes here :) - Thanks Rachael, for all the groundwork ! :-)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# Query1 = bitcoin transactions done per day, month after month in 2017 :-)
# Query2 = consolidated bitcoin transactions done per day in 2017
# Query3 = transactions done per merkle root irrespective of year
# Query4 = transactions done per merkle root in year 2017

query1 = """ WITH day4_bitcoin_trans AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id)       AS transactions,
                EXTRACT(DAY FROM trans_time)   AS day  ,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time)  AS year
            FROM day4_bitcoin_trans
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day, month, year 
            ORDER BY day, month, year
        """
# note that max_gb_scanned is set to 21, rather than 1
q1trans_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
display(q1trans_per_day)

query2 = """ WITH day4_bitcoin_trans AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id)       AS transactions,
                EXTRACT(DAY FROM trans_time)   AS day  ,
                EXTRACT(YEAR FROM trans_time)  AS year
            FROM day4_bitcoin_trans
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day, year
            ORDER BY day, year
        """
# note that max_gb_scanned is set to 21, rather than 1
q2trans_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
display(q2trans_per_day)

query3 = """ SELECT COUNT(transaction_id) AS transactions,
                    merkle_root
             FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             GROUP BY merkle_root 
             ORDER BY merkle_root
         """
# note that max_gb_scanned is set to 50, rather than 21
q3trans_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=50)
display(q3trans_per_merkle)

query4 = """ WITH day4_bitcoin_trans AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id,
                    merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id)       AS transactions,
                EXTRACT(YEAR FROM trans_time)  AS year,
                merkle_root
            FROM day4_bitcoin_trans
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, merkle_root 
            ORDER BY year, merkle_root
        """
# note that max_gb_scanned is set to 50, rather than 21
q4trans_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query4, max_gb_scanned=50)
display(q4trans_per_merkle)

