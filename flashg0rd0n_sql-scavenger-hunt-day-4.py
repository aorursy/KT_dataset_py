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
# Query how many Bitcon tranactions happened per day in 2017
Q1 = """ WITH time AS 
(
        SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        WHERE EXTRACT(YEAR FROM (TIMESTAMP_MILLIS(timestamp))) = 2017
)

     SELECT 
        DATE(EXTRACT(YEAR FROM trans_time), 
        EXTRACT(MONTH FROM trans_time), 
        EXTRACT(DAY FROM trans_time)) AS date,
        COUNT(transaction_id) AS transactions

    FROM time
    GROUP BY date
    ORDER BY date
    """

# transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(Q1, max_gb_scanned=21)
transactions_per_month = bitcoin_blockchain.query_to_pandas(Q1)
print("Number of transactions per day on Bitcoin")
transactions_per_month

# plot bitcoin transactions per day in 2017
plt.plot(transactions_per_month.date, transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Query of transactions associated with each merkle root
Q2 = """ WITH time AS 
(
        SELECT merkle_root, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)

     SELECT 
        merkle_root,
        COUNT(transaction_id) AS transactions

    FROM time
    GROUP BY merkle_root
    ORDER BY transactions DESC
    """
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas(Q2)
transactions_per_merkle_root.head()