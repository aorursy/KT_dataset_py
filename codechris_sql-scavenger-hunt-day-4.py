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
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                   transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`                
            )
            
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day
                
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day
            ORDER BY day
            """


trans_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
print(trans_day)
import matplotlib.pyplot as plt

plt.plot(trans_day)
query = """ SELECT COUNT(transaction_id) as transactions,
                merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions DESC
            """
merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
merkle

plt.plot(merkle.transactions)