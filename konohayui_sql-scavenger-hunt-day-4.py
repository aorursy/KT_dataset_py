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
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day
            ORDER BY day
        """

transaction_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 21)
fig = plt.figure(figsize = (15, 10))
plt.plot(transaction_per_day_2017.transactions)
plt.xlabel("Day", fontsize = 15)
plt.ylabel("Transactions", fontsize = 15)
plt.title("Bitcoin Transaction Per Day In 2017", fontsize = 20)
plt.show()
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY month
            ORDER BY month
        """

transaction_per_month_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 21)
import seaborn as sns

fig = plt.figure(figsize = (15, 10))
sns.barplot(transaction_per_month_2017.month, transaction_per_month_2017.transactions)
plt.xlabel("Month", fontsize = 15)
plt.ylabel("Transactions", fontsize = 15)
plt.title("Total Transaction Per Month In 2017", fontsize = 20)
plt.show()
query = """ WITH merkles AS 
            (
                SELECT merkle_root,
                    COUNT(transaction_id) as transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
            )
            SELECT merkle_root, transactions
            FROM merkles
            WHERE transactions > 5000
            ORDER BY transactions DESC
        """

transactions_associate_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 37)
transactions_associate_merkle