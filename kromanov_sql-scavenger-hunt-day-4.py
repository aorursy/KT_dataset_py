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
# Your code goes here :)
# How many Bitcoin transactions were made each day in 2017?
query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY day, month, year
            HAVING year=2017
            ORDER BY day, month, year
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)

plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
bitcoin_blockchain.head('transactions')
# Your code goes here :)
# How many transactions are associated with each merkle root
query3 = """ WITH transaction_nb AS 
            (  
                SELECT COUNT(transaction_id) AS transactions,  
                merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
            )
            SELECT *
            FROM transaction_nb
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=37)
n, bins, patches = plt.hist(transactions_per_merkle_root.transactions, 100)
plt.xlim(0, 1000)
plt.title("Number of transactions associated with each merkle root")
plt.show()