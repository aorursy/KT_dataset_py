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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

#How many Bitcoin transactions were made each day in 2017?
transactions_per_day_query = """ WITH view_transactions AS 
            (
                SELECT 
                    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS trans_year,
                    EXTRACT(DATE FROM TIMESTAMP_MILLIS(timestamp)) AS trans_date,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT trans_date, count(*)
            FROM view_transactions
            WHERE trans_year = 2017
            GROUP BY transaction_id, trans_date
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas(transactions_per_day_query)
print (transactions_per_day)

#How many transactions are associated with each merkle root?
transaction_merkleroot_association_query = """ SELECT merkle_root, count(transaction_id)
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
        """
transaction_merkleroot_association = bitcoin_blockchain.query_to_pandas(transaction_merkleroot_association_query)
print (transaction_merkleroot_association)