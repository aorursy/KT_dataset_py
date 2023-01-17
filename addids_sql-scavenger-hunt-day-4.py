# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")


#Q1: How many Bitcoin transactions were made each day in 2017?
#You can use the "timestamp" column from the "transactions" table to 
#answer this question. You can check the notebook from Day 3 for more information on timestamps.

query = """ WITH time AS 
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
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day 
            ORDER BY year, month, day
        """


bitcoin_blockchain.estimate_query_size(query)
# note that max_gb_scanned is set to 21, based on the result above
transactions_in_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_in_2017.transactions)
plt.title("Daily Bitcoin Transactions (2017)")
#Q2: How many blocks are associated with each merkle root?
#You can use the "merkle_root" and "block_id" columns in 
#the "blocks" table to answer this question.

query = """ SELECT COUNT(block_id) AS nb_block, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            GROUP BY merkle_root 
            ORDER BY nb_block
        """
bitcoin_blockchain.estimate_query_size(query)
nb_blocks = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=1)
nb_blocks.head()