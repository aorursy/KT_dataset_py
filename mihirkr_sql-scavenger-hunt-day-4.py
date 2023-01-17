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
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DATE FROM trans_time) AS day
                
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time)  = 2017
            GROUP BY day
            ORDER BY day
        """

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot daily bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
query2 = """
select merkle_root, count(transaction_id) as transactions
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
"""

transactions_by_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
# plot bitcoin transactions by merkle_root
plt.plot(transactions_by_merkle.transactions)
plt.title("Bitcoin Transcations by merkle_root")