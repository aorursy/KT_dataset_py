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
#transactions_per_month
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Question 2:
#bitcoin_blockchain.table_schema("transactions")
query = """
    SELECT count(transaction_id) AS trans_per_merkle, merkle_root
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle_root
    ORDER BY merkle_root
"""
test = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
test["trans_per_merkle"].hist(log=True)
plt.xlabel("Number of Transactions")
plt.ylabel("Number of Merkles (Log)")
plt.show()
# Question 1:
query = """ 
            WITH days AS
            (
                WITH time AS 
                (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                        transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
            SELECT transaction_id,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            )
            SELECT COUNT(transaction_id) AS trans_count, month, day
            FROM days
            WHERE year = 2017
            GROUP BY month, day
            ORDER BY month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
test = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
plt.plot(test.trans_count)
plt.title("Transaction per-day during 2017")
plt.show()