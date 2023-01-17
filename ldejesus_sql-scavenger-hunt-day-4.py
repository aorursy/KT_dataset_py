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
# How many transactions where made each day in 2017

query_day = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY month, day
            ORDER BY month, day
        """
bitcoin_blockchain.estimate_query_size(query_day)
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_day, max_gb_scanned=21)
transactions_per_day
# Plot daily transactions
import matplotlib.pyplot as plt

# plot daily bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
# How many transactions are associated with each merkle root?
query_merkle = """    
                SELECT merkle_root,
                       COUNT(transaction_id) as trans_count
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`                    
                GROUP BY merkle_root
                ORDER BY trans_count DESC
              """
bitcoin_blockchain.estimate_query_size(query_merkle)
# note that max_gb_scanned is set to 21, rather than 1
merkle_trans = bitcoin_blockchain.query_to_pandas_safe(query_merkle, max_gb_scanned=37)
merkle_trans
# plotting library is already imported
#import matplotlib.pyplot as plt

# plot transactions associated with Merkle
plt.plot(merkle_trans.trans_count)
plt.title("Transactions associated with Merkle")