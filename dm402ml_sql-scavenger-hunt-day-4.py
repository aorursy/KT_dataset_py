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
# Daniela's code goes here :)
#1-How many Bitcoin transactions were made each day in 2017?
query1 = """ WITH trans_day AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(day FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM trans_day
            GROUP BY year,month,day 
            ORDER BY year,month,day
        """

# note that max_gb_scanned is set to 25, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=25)
#Printing results
print('List of 2017 daily Bitcoins transactions')
print (transactions_per_day)

# plot daily bitcoin transactions 2017
plt.plot(transactions_per_day.transactions)
plt.title("2017 Daily Bitcoin Transactions")

#2-How many transactions are associated with each merkle root?
query2 = """ WITH trans_merkle_root AS 
            (
                SELECT merkle_root as mk,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT mk,
                COUNT(transaction_id) AS transactions
            FROM trans_merkle_root
            GROUP BY mk 
            ORDER BY transactions DESC
        """

# note that max_gb_scanned is set to 37, rather than 25
transactions_mk = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)
#Printing results
print('List of transactions associated to each merkle root')
print (transactions_mk)