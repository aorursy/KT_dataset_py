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
                EXTRACT(DATE FROM trans_time) AS date,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, date
            ORDER BY year, date
            
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_date = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=25)

print(transactions_per_date)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_date.transactions)
plt.title("Daily Bitcoin Transcations")
query = """ WITH transactions AS 
            (
                SELECT transaction_id, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions
            FROM transactions
            GROUP BY merkle_root
            ORDER BY merkle_root
            
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)

print(transactions_merkle)