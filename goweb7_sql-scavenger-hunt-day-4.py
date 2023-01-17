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
query_trans = """
              WITH transaction_list AS
                  (
                      SELECT TIMESTAMP_MILLIS(timestamp) AS timestamp,
                             transaction_id
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                      WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
                  )
              SELECT DATE(timestamp) AS Date,
                     COUNT(transaction_id) AS Transactions
              FROM transaction_list
              GROUP BY Date
              ORDER BY Date ASC
              """

trans_df = bitcoin_blockchain.query_to_pandas_safe(query_trans, max_gb_scanned=21)
print('Bitcoin transactions in 2017')
print(trans_df.head().to_string(index=False, justify='center'))

print()

query_merkle = """
               WITH transaction_list AS
                   (
                       SELECT merkle_root,
                              block_id
                       FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                   )
               SELECT merkle_root AS Root,
                      COUNT(block_id) AS Blocks
               FROM transaction_list
               GROUP BY Root
               ORDER BY Blocks DESC
               """

merkle_df = bitcoin_blockchain.query_to_pandas_safe(query_merkle, max_gb_scanned=40)
print('Blocks by merkle root')
print(merkle_df.head().to_string(index=False, justify='center'))