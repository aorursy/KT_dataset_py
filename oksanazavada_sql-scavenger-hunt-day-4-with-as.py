# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

# print the first couple rows of the "transactions" dataset
bitcoin_blockchain.head("transactions")

query_task1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
                WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day 
            ORDER BY day
            """

query_task2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT merkle_root as merkle_root, COUNT(transaction_id) AS blocks
                FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
                GROUP BY merkle_root 
                ORDER BY merkle_root
            """

transactions_per_day = bitcoin_blockchain.query_to_pandas(query_task1)

transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas(query_task2)

# import plotting library
import matplotlib.pyplot as plt

# plot dayly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Dayly Bitcoin Transcations in 2017")
plt.show()
print(transactions_per_day)

print(transactions_per_merkle_root)
