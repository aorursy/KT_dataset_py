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
# How many Bitcoin transactions were made each day in 2017?
day_2017_query = """
WITH time_stamp AS
(SELECT TIMESTAMP_MILLIS(timestamp) as transaction_stamp,
transaction_id
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT COUNT(transaction_id) AS count,
EXTRACT(DAY FROM transaction_stamp) AS day
FROM time_stamp
WHERE EXTRACT(YEAR FROM transaction_stamp)=2017
GROUP BY day
ORDER BY day
"""

day_transaction_set = bitcoin_blockchain.query_to_pandas_safe(day_2017_query, max_gb_scanned=21)
print(day_transaction_set)

# How many transactions are associated with each merkle root?
merkle_transaction_query = """
SELECT COUNT(transaction_id) AS transactions,
merkle_root AS merkle
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle
"""

merkle_transaction_set = bitcoin_blockchain.query_to_pandas_safe(merkle_transaction_query, max_gb_scanned=37)
merkle_transaction_set.head()