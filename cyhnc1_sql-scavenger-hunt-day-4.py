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
bitcoin_blockchain.head("transactions")
# this query will return total number of daily transactions in 2017.
query2 = """with time as
            (
                select timestamp_millis(timestamp) as trans_time,
                        transaction_id
                from `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            select extract(year from trans_time) as year,
                   extract(month from trans_time) as month,
                   extract(day from trans_time) as day,
                   count(transaction_id) as total_transactions
            from time
            where extract(year from trans_time) = 2017
            group by year, month, day
            order by year, month, day
            """
# let's get the estimate for the size of the data need to be scanned 
bitcoin_blockchain.estimate_query_size(query2)
# store output in a pandas dataframe
daily_transactions = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned = 21)
daily_transactions.shape
daily_transactions.head(10)
# return the number of transactions associated with each merkle root.
query3 = """select merkle_root as merkle_root,
                   count(transaction_id) as total_transactions
            from `bigquery-public-data.bitcoin_blockchain.transactions`
            group by merkle_root
            order by total_transactions desc"""

bitcoin_blockchain.estimate_query_size(query3)
transactions_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned = 37)
transactions_by_merkle_root.shape
transactions_by_merkle_root.head(10)