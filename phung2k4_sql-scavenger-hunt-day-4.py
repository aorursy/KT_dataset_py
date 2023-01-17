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
"""
How many Bitcoin transactions were made each day in 2017?
"""

query1 = """ 
            WITH time AS 
                (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                ),
            extract_time_info AS 
                (
                    SELECT transaction_id,
                        EXTRACT(MONTH FROM trans_time) AS Month,
                        EXTRACT(YEAR FROM trans_time) AS Year,
                        EXTRACT(DAY FROM trans_time) AS Day
                    FROM time
                )
                SELECT Month, Day, COUNT(transaction_id) AS Transactions
                FROM extract_time_info 
                WHERE Year = 2017
                GROUP BY Month, Day
                ORDER BY Month, Day
        """
# check how big this query will be
# bitcoin_blockchain.estimate_query_size(query1)
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)

print(transactions_per_day.head())
"""
How many transactions are associated with each merkle root?
"""

query2 = """
        SELECT merkle_root, COUNT(transaction_id) AS Transactions
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        """

# check how big this query will be
# bitcoin_blockchain.estimate_query_size(query2)
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=38)
print(transactions_per_merkle_root.head(n=10))