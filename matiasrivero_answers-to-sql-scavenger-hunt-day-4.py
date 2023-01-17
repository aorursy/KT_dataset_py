# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# query to see how many Bitcoin transactions were made each day in 2017
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

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.figure(figsize=(12,6))
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transactions in 2017")
plt.xlabel('Day in 2017 (from 1 to 366)')
plt.ylabel('Number of Transactions')
# query to see how many transactions are associated with each merkle root
query2 = """ WITH merkle AS 
            (
                SELECT merkle_root, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) as Transactions, merkle_root as Merkle_Root
            FROM merkle
            GROUP BY merkle_root
            ORDER BY transactions DESC
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
# number of transactions by merkel_root, ordered in a descent way
transactions_per_merkle.head(20)
