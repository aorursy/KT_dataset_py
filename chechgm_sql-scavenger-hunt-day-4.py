# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# Now let's check the tables the database has
bitcoin_blockchain.list_tables()
# And the head of transactions
bitcoin_blockchain.head("transactions")
# And let's check a bit of extra information on the table
bitcoin_blockchain.table_schema("transactions")
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
# Query 1
# How many Bitcoin transactions were made each day in 2017?

# We write the Query
query_1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY month, day
            ORDER BY month, day
        """

# We run the query 
transactions_per_day = bitcoin_blockchain.query_to_pandas(query_1)

# We check the results
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transactions")
# Query 2
# How many transactions are associated with each merkle root?

# We write the Query
query_2 = """ WITH merkle AS 
            (
                SELECT transaction_id, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM merkle
            GROUP BY merkle_root
        """

# We run the query 
transactions_per_merkle = bitcoin_blockchain.query_to_pandas(query_2)

# We check the results
transactions_per_merkle.head()