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
# How many Bitcoin transactions were made each day in 2017

query = """ WITH time AS
            ( 
               SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                   transaction_id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )  
               SELECT EXTRACT(DAY FROM trans_time) AS day,
                      EXTRACT(MONTH FROM trans_time) AS month,
                      COUNT(transaction_id) AS transactions
                  
             FROM time
             WHERE EXTRACT( YEAR FROM trans_time) = 2017
             GROUP BY month, day 
             ORDER BY month, day
            
        """

# estimate query size 
bitcoin_blockchain.estimate_query_size(query)
# max gb scanned is set to 21 rather than 1 

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 21)

# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Monthly Bitcoin Transcations \n(Per Day in the year 2017)")
# Results 
transactions_per_day
# How many transactions are associated with each merkle root 

query = """ WITH trans AS
           (
               SELECT transaction_id , merkle_root
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
            SELECT merkle_root, COUNT(transaction_id) AS transactions
            FROM trans
            GROUP BY merkle_root
        """
# estimate the size of the query 

bitcoin_blockchain.estimate_query_size(query)
# Max_gB_scanned is set to 37 , rather than 1 

transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 37)
# results 
transactions_per_merkle