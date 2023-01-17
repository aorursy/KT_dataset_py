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
query1 = """WITH subtable AS(
        SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id) AS transactions, EXTRACT(DAY FROM trans_time) AS day,
            EXTRACT(MONTH FROM trans_time) AS month
        FROM subtable
        WHERE EXTRACT(YEAR FROM trans_time) = 2017
        GROUP BY day, month
        ORDER BY month, day
"""

df1 = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
#df1
plt.plot(df1["transactions"])
query2 = """WITH subtable AS(
        SELECT merkle_root, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id) AS transactions, merkle_root
        FROM subtable
        GROUP BY merkle_root
        ORDER BY transactions DESC
"""

df2 = bitcoin_blockchain.query_to_pandas(query2)

df2.head()