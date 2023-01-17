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
# print all the tables in this dataset
bitcoin_blockchain.list_tables()
# print the first couple rows of the "transactions" table
bitcoin_blockchain.head("transactions")
query_yearly_transaction = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year
            ORDER BY year
        """
# check how big this query will be
bitcoin_blockchain.estimate_query_size(query_yearly_transaction)
yearly_transactions = bitcoin_blockchain.query_to_pandas_safe(query_yearly_transaction, max_gb_scanned=21)
yearly_transactions
query_merkleroot_transaction = """ WITH merkleroot AS 
            (
                SELECT transaction_id,
                    merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM merkleroot
            GROUP BY merkle_root
            ORDER BY COUNT(transaction_id) desc
        """
# check how big this query will be
bitcoin_blockchain.estimate_query_size(query_merkleroot_transaction)
merkleroot_transactions = bitcoin_blockchain.query_to_pandas_safe(query_merkleroot_transaction, max_gb_scanned=37)
merkleroot_transactions