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
query2 = """
        WITH time as
        (
            SELECT DATE(TIMESTAMP_MILLIS(timestamp)) AS correctTime,
                transaction_id AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        
        SELECT correctTime AS Day, COUNT(transactions) AS DailyTransactions
        FROM time
        WHERE EXTRACT(YEAR FROM correctTime) = 2017
        GROUP BY Day
        ORDER BY Day
        """

BIT_Per_Day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)

BIT_Per_Day.head()
# Import matplotlib

import matplotlib.pyplot as plt

# Plot the information we have gathered

plt.plot(BIT_Per_Day.DailyTransactions)
plt.title("Daily Bitcoin Transactions")
query3 = """
        SELECT merkle_root, count(transaction_id) as num_tra
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        ORDER BY num_tra DESC
        """

merkles = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned = 37)

print(merkles)