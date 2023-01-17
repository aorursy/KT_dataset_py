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
transactions_per_month.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
query2 = """ 
                    WITH time AS 
                (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, 
                    transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(transaction_id) AS transactions,
                    EXTRACT(DAY FROM trans_time) AS day,
                    EXTRACT(MONTH FROM trans_time) AS month,
                    EXTRACT(YEAR FROM trans_time) AS year
                FROM time
               
                GROUP BY year, month, day
                HAVING year=2017
                ORDER BY year, month, day
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
transactions_per_day.head()
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
# How many transactions are associated with each merkle root?
import bq_helper

bitcoin_blockchain_trans = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="transactions")

query3 = """ WITH merkle AS 
            (
                SELECT merkle_root, count(transaction_id) as transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
                ORDER BY transactions DESC
            )
            SELECT merkle_root, transactions
            FROM merkle
        """
trans_merkle = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=37)
trans_merkle.head()