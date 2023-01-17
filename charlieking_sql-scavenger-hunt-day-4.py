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
## How many Bitcoin transactions were made each day in 2017?
## Set up our environment
import bq_helper
## Create helper object
bitcoin = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="bitcoin_blockchain")
## Have a look at the head of transactions
bitcoin.head(table_name="transactions")
## Query - Remember we cannot refer to an aliased field in the WHERE statement because
## WHERE gets executed before the SELECT statement
query = """WITH cte AS
        (
            SELECT EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) as year,
            EXTRACT(MONTH FROM TIMESTAMP_MILLIS(timestamp)) as month,
            EXTRACT(DAY FROM TIMESTAMP_MILLIS(timestamp)) as day, transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )

        SELECT  year, month, day,
                COUNT(transaction_id) AS n_transactions
        FROM cte
        WHERE  year = 2017
        GROUP BY year,month, day;"""
bitcoin.estimate_query_size(query)
## Query to pandas
df = bitcoin.query_to_pandas(query)
## Show first rows
df.head(35)
## Little plot
df["n_transactions"].plot(title="Number of BTC Transactions per day in 2017")
## How many transactions are associated to each merkle_root?
## Query
query = """
SELECT merkle_root, COUNT(transaction_id) AS n_transactions 
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY n_transactions DESC;
"""
## Query to Pandas
df = bitcoin.query_to_pandas(query)
## Plot DF's head
df.head()
