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
# Displaying the .head() of the working table
bitcoin_blockchain.head("transactions")
# Setting the QUERYs

QUERY1 = """WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id,
                    EXTRACT(DAYOFYEAR FROM TIMESTAMP_MILLIS(timestamp)) AS day,
                    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS year
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions_count, day, year
            FROM time
            WHERE year = 2017
            GROUP BY year, day
            ORDER BY transactions_count DESC
        """

QUERY2 = """WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id, merkle_root,
                    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS year
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transaction_count, merkle_root
            FROM time
            WHERE year = 2017
            GROUP BY merkle_root
            ORDER BY transaction_count DESC
        """   
# SOlving question 1
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(QUERY1, max_gb_scanned=21)
# Displaying the QUERY result
transactions_per_day.head()
# Sorting by day and plotting transactions each day of 2017
transactions_per_day.sort_values(by='day').plot(x='day', y='transactions_count', figsize=(12,8));
plt.title('Transaction by day in 2017');
# SOlving question 2
# note that max_gb_scanned is set to 21, rather than 1
count_per_root = bitcoin_blockchain.query_to_pandas_safe(QUERY2, max_gb_scanned=40)
# Displaying the QUERY result
count_per_root.head()