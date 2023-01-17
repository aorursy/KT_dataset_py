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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
#Let's create a CTE for our first query and find the number of daily transactions for the year 2017.
query1 = """WITH time_new AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) as Time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS Transactions_Count,
                EXTRACT(DAY FROM Time) AS Day, EXTRACT(MONTH FROM Time) AS Month
            FROM time_new
            WHERE EXTRACT(YEAR FROM Time) = 2017
            GROUP BY Month,Day
            ORDER BY Month,DAY
            """

transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
transactions_2017

#For better understanding, let us plot it.
import matplotlib.pyplot as plt
plt.plot(transactions_2017)
plt.title("Transactions per day for 2017")
plt.xlabel("Days")
plt.ylabel("Frequency")

#For the second query
query2 = """SELECT COUNT(transaction_id) AS Transactions, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY Transactions DESC
            """
merkle_transaction = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)
merkle_transaction