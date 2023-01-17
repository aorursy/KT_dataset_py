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
query = """WITH time AS (
SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
transaction_id
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
WHERE EXTRACT (YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
    )
SELECT COUNT(transaction_id) AS transactions,
    EXTRACT(DATE FROM trans_time) AS Date
    FROM time
    GROUP BY Date
    ORDER BY transactions
"""

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

# transactions_per_day.head()

import matplotlib.pyplot as plt

# plot daily bitcoin transactions
plt.bar(transactions_per_day.Date,transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
query = """ WITH trans AS (
SELECT DISTINCT transaction_id,
merkle_root
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT COUNT(transaction_id) AS transactions,
    merkle_root
    FROM trans 
    GROUP BY merkle_root
    ORDER BY transactions desc
"""

# bitcoin_blockchain.estimate_query_size(query)
transactions_per_merkel = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)

transactions_per_merkel.head()

# plt.bar(transactions_per_merkel.merkle_root,transactions_per_merkel.transactions)
# plt.title("Bitcoin Transcations per Merkel Root")