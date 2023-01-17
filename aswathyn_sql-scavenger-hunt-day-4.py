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
# Scavenger Hunt
# How many Bitcoin transactions were made each day in 2017?
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                   EXTRACT(DAY FROM trans_time) AS day 
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time)=2017
            GROUP BY day
            ORDER BY transactions DESC
        """

transaction_per_day2017=bitcoin_blockchain.query_to_pandas_safe(query1,max_gb_scanned=25)
transaction_per_day2017

# How many Bitcoin transactions were made each day in 2017?
query2 = """ WITH merkels AS (
                  SELECT merkle_root,
                         transaction_id
                  FROM `bigquery-public-data.bitcoin_blockchain.transactions`
               )
             SELECT merkle_root,COUNT(transaction_id ) AS no_of_transactions 
             FROM merkels 
             GROUP BY merkle_root 
             ORDER BY no_of_transactions DESC"""

bitcoin_blockchain.estimate_query_size(query2)

transaction_per_merkle=bitcoin_blockchain.query_to_pandas_safe(query2,max_gb_scanned=37)
transaction_per_merkle