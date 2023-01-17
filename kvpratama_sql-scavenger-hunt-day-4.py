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
query_day = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            GROUP BY day 
            ORDER BY day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_day, max_gb_scanned=21)

transactions_per_day
plt.plot(transactions_per_day.day, transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
plt.xlabel("Date")
plt.ylabel("Volume")
query_dayofweek = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFWEEK FROM trans_time) AS dayofweek
            FROM time
            GROUP BY dayofweek
            ORDER BY dayofweek
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_dayofweek = bitcoin_blockchain.query_to_pandas_safe(query_dayofweek, max_gb_scanned=21)
plt.bar(transactions_per_dayofweek.dayofweek, transactions_per_dayofweek.transactions)
plt.title("Day of Week Bitcoin Transcations")
plt.xlabel("Day of Week")
plt.ylabel("Volume")
query_merkle_root = """
            SELECT COUNT(transaction_id) AS transactions_no,
                merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions_no
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query_merkle_root, max_gb_scanned=37)
transactions_per_merkle_root