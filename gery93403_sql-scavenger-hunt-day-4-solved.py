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
#How many Bitcoin transactions were made each year in 2017?
query_per_day = """
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

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_per_day, max_gb_scanned=21)

transactions_per_day.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day.DailyTransactions)
plt.title("Daily Bitcoin Transcations")
# How many transactions are associated with each merkle root?

query_per_merkle_root = """ SELECT merkle_root AS merkle, COUNT(transaction_id) as transactions
FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
GROUP BY merkle_root
ORDER BY transactions DESC"""


transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query_per_merkle_root, max_gb_scanned=37)

transactions_per_merkle.head()