# Your code goes here :)

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT COUNT(transaction_id) AS transactions, 
            EXTRACT(DATE FROM trans_time) AS date        
            FROM time            
            GROUP BY date 
            ORDER BY date ASC
        """

# note that max_gb_scanned is set to 21, rather than 1
day_transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(day_transactions.date, day_transactions.transactions)
plt.title("Bitcoin Transactions Count per day")
# add weekday to df
day_transactions['weekday'] = day_transactions['date'].apply(lambda x: x.weekday())

# aggregate by day
day_transactions.groupby(day_transactions.weekday, as_index=False).agg({"transactions": "sum"})
query = """ WITH block_table AS 
            (
                SELECT merkle_root, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`                
            )
            SELECT merkle_root, COUNT(transaction_id) AS transactions             
            FROM block_table            
            GROUP BY merkle_root
            HAVING transactions > 0
            ORDER BY transactions DESC
        """

# keep max_gb_scanned set to 40, estimated > 36.8gb
merkle_root_blocks = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)

# Only showing 10 of 509133 results
merkle_root_blocks.head(10)