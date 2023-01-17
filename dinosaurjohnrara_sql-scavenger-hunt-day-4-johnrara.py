# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# print the first couple rows of the "blocks" table- transposed 
bitcoin_blockchain.head("blocks").transpose()
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
# print the first couple rows of the "transactions" table- transposed 
bitcoin_blockchain.head("transactions").transpose()

#How many Bitcoin transactions were made each day in 2017?
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(YEAR FROM trans_time) AS year,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            GROUP BY year, month, day
            HAVING year = 2017 
        """
#query returns a dataframe
blockchain_transactions_2017 = bitcoin_blockchain.query_to_pandas(query1)
blockchain_transactions_2017
#How many transactions are associated with each merkle root?
#You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question.
query2 = """SELECT merkle_root, COUNT(transaction_id)
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        """
#query returns a dataframe
merkle_root_transactions = bitcoin_blockchain.query_to_pandas(query2)
merkle_root_transactions