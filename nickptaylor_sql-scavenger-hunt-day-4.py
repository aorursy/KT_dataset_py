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
query = """WITH time AS (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT EXTRACT(DAYOFYEAR from trans_time) AS yearday,
                  COUNT(transaction_id) as transactions
               FROM time
               GROUP BY yearday
               ORDER BY yearday
"""

# Check size of query.
bitcoin_blockchain.estimate_query_size(query)

# Run query.
df = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 25)
my_plot = df.plot(y = 'transactions', x = 'yearday')
my_plot.legend().set_visible(False)
my_plot.set_xlabel("Day of year")
my_plot.set_ylabel("Transactions")
query = """SELECT count(block_id) AS blocks, merkle_root
               FROM `bigquery-public-data.bitcoin_blockchain.blocks`
               GROUP BY merkle_root
               ORDER BY blocks DESC
"""

# Check size of query.
bitcoin_blockchain.estimate_query_size(query)

# Run query.
df = bitcoin_blockchain.query_to_pandas_safe(query)
df
# Your code goes here :)

