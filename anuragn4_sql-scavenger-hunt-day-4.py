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
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head("transactions")

query = """ WITH TIME AS
              (
                SELECT TIMESTAMP_MILLIS(timestamp) AS tran_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
              )
              
              SELECT EXTRACT(YEAR from tran_time) AS YEAR,
                  COUNT(transaction_id) AS NUM_TRANSACTIONS
              FROM TIME
              GROUP BY YEAR
              HAVING YEAR=2017    
        """
transactions_per_year = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_year

query1 = """ 
             SELECT merkle_root, COUNT(transaction_id) as NUM_TRANSACTIONS
             FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             GROUP BY merkle_root
             ORDER BY NUM_TRANSACTIONS
         """
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=37)
transactions_per_merkle_root