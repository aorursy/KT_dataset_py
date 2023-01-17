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
# form query for question 1
query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            ),
            transactions2017 AS
            (
                SELECT COUNT(transaction_id) AS transactions,
                    EXTRACT(DAYOFWEEK FROM trans_time) AS day,
                    EXTRACT(YEAR FROM trans_time) AS year
                FROM time
                GROUP BY year, day 
                HAVING year=2017
                ORDER BY year,day
            )
            SELECT day,transactions
            FROM transactions2017
            WHERE year=2017
            ORDER BY day
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)

transactions_per_day.head()
# make a plot to show that our data is, actually, sorted:
plt.bar(transactions_per_day.day,transactions_per_day.transactions)
plt.title("Transactions in bitcoins \n per day of the week in 2017")
plt.ylabel("Transactions")
plt.xticks(transactions_per_day.day, ['Mon', 'Tues','Wed','Thur','Fri','Sat','Sun'])
plt.xlabel("Day of the week")
# form query for question 2
query3 = """ 
            SELECT merkle_root, COUNT(transaction_id) as transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions DESC
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=37)
transactions_per_merkle.head()
transactions_per_merkle.to_csv("transactions_per_merkle.csv")
