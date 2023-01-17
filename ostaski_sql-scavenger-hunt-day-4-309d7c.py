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
# importing the Big Query helper package
import bq_helper

# create the helper object using bigquery-public-data.bitcoin_blockchain
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                              dataset_name="bitcoin_blockchain")

# looking at the transactions data
bitcoin_blockchain.list_tables() 
bitcoin_blockchain.table_schema("transactions")
bitcoin_blockchain.head("transactions")

# Q1. How many Bitcoin transactions were made each day in 2017?
query = """ WITH time AS 
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
            ORDER BY transactions
        """

# estimate query size
bitcoin_blockchain.estimate_query_size(query)
# 20.633303198963404, so need to bump the max_gb_scanned in query up to 21 gb

# run a "safe" query and store the resultset into a dataframe
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

# taking a look
print(transactions_per_day_2017)
# day 14 had the most transaction and day 31 had the least

# saving this in case we need it later
transactions_per_day_2017.to_csv("transactions_per_day_2017.csv")

# Q2. How many transactions are associated with each merkle root?

# the query
query = """ SELECT merkle_root, 
                COUNT(transaction_id) AS transactions_count 
            FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
            GROUP BY merkle_root
            ORDER BY transactions_count
        """

# estimate query size
bitcoin_blockchain.estimate_query_size(query)
# 36.82227347418666, so bump the max_gb_scanned in query up to 37 gb

# run a "safe" query and store the resultset into a dataframe
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)

# taking a look
print(transactions_per_merkle_root)
# transactions_count ranges from 1 to 12239 over 509278 merkle_roots

# saving this in case we need it later
transactions_per_merkle_root.to_csv("transactions_per_merkle_root.csv")
