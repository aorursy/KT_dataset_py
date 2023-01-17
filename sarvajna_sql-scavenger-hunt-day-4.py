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
bitcoin_blockchain.table_schema(table_name="transactions")
bitcoin_blockchain.head(table_name="transactions")
#How many Bitcoin transactions were made each day in 2017
query1 = """WITH time_cte AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) as number_of_transactions, 
                EXTRACT(YEAR FROM transaction_time) AS year_of_transaction,
                EXTRACT(DAYOFYEAR FROM transaction_time) AS day_of_transaction
            FROM time_cte
            GROUP BY year_of_transaction, day_of_transaction
            HAVING year_of_transaction = 2017
            ORDER BY day_of_transaction
         """

print(bitcoin_blockchain.estimate_query_size(query=query1))

transactions_per_day_2017_df = bitcoin_blockchain.query_to_pandas_safe(query=query1, 
                                                                       max_gb_scanned=21)           
plt.plot(transactions_per_day_2017_df.number_of_transactions)
plt.title("Daily bitcoin transactions")
transactions_per_day_2017_df.head(n=366)
# How many transactions are associated with each merkle root?
query2 = """SELECT COUNT(transaction_id) AS number_of_transactions, merkle_root 
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY number_of_transactions DESC
         """

print(bitcoin_blockchain.estimate_query_size(query=query2))

transactions_per_merkle_root_df = bitcoin_blockchain.query_to_pandas_safe(query=query2, 
                                                                       max_gb_scanned=37)
transactions_per_merkle_root_df.head()