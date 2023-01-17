# Your code goes here :)
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
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day 
            ORDER BY year, month, day
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day.head()
#graph
# import plotting library
import matplotlib.pyplot as plt
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
query="""SELECT COUNT(block_id) AS numBlocks, merkle_root
           FROM `bigquery-public-data.bitcoin_blockchain.blocks`
           GROUP BY merkle_root
           ORDER BY numBlocks
        """
merkleBlocks = bitcoin_blockchain.query_to_pandas(query)
merkleBlocks.head()
query="""SELECT COUNT(transaction_id) AS numTransactions, merkle_root
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY merkle_root
           ORDER BY numTransactions DESC
        """
merkleTransactions = bitcoin_blockchain.query_to_pandas(query)
merkleTransactions.head()