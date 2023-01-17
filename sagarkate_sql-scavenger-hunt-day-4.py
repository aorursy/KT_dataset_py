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
# Your code goes here :)
#How many Bitcoin transactions were made each day in 2017?
queryTransactionsPerDay2017 = """
WITH dayTransactions AS (
SELECT
TIMESTAMP_MILLIS(timestamp) as transaction_time,
transaction_id
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT 
EXTRACT(DAYOFYEAR FROM transaction_time) as dayofyear,
COUNT(1) as no_of_transactions
FROM dayTransactions
WHERE EXTRACT(YEAR FROM transaction_time) = 2017
GROUP BY dayofyear
ORDER BY dayofyear
"""
resultTransactionsPerDay2017 = bitcoin_blockchain.query_to_pandas_safe(queryTransactionsPerDay2017, max_gb_scanned=3)
print("No. of transactions per day in year 2017:")
print(resultTransactionsPerDay2017)

#How many transactions are associated with each merkle root?
queryTransactionsPerMerkleRoot = """
SELECT
merkle_root,
COUNT(1) as no_of_transactions
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY no_of_transactions DESC
"""

resultTransactionsPerMerkleRoot = bitcoin_blockchain.query_to_pandas(queryTransactionsPerMerkleRoot)
print("No. of transactions per merkle root:")
print(resultTransactionsPerMerkleRoot)