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
import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
											  
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day
            FROM time
			WHERE EXTRACT(YEAR FROM trans_time)=2017
            GROUP BY day
            ORDER BY day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day_2017.head()

transactions_per_day_2017
import matplotlib.pyplot as plt
plt.plot(transactions_per_day_2017)
plt.title("Daily Bitcoin Transactions")
# How many transactions are associated with each merkle root?
#   --> merkle
query = """ SELECT merkle_root,
                    count(transaction_id) as nb_transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
				group by merkle_root
        """
transactions_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
transactions_merkle

# count merkle_root ...
query = """ WITH merkle AS 
            (
                SELECT merkle_root,
                    count(transaction_id) as nb_transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
				group by merkle_root
            )
            SELECT min(nb_transactions),max(nb_transactions),count(merkle_root),sum(nb_transactions)
            FROM merkle
        """
transactions_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
transactions_merkle