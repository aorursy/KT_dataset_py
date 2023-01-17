import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
            SELECT EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                   COUNT(transaction_id) AS transactions
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day 
            ORDER BY day
        """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
transactions_per_day
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations for 2017")
query2 = """ SELECT merkle_root,
                   COUNT(transaction_id) AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY 2 DESC
        """
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)
transactions_per_merkle_root
plt.plot(transactions_per_merkle_root.transactions)
plt.title("Transactions per Merkle Root")
query3 = """ WITH mr_trans AS (
             SELECT merkle_root,
                   COUNT(transaction_id) AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            )
            , mr_grouped AS (
             SELECT 1 AS grOrder, '> 10,000' AS MerkleGroup, COUNT( merkle_root) AS GroupCount
             FROM mr_trans
             WHERE transactions >= 10000
             UNION ALL
             SELECT 2, '7,500 - 10,000', COUNT( merkle_root)
             FROM mr_trans
             WHERE transactions BETWEEN 7500 AND 9999
             UNION ALL
             SELECT 3, '5,000 - 7,500', COUNT( merkle_root)
             FROM mr_trans
             WHERE transactions BETWEEN 5000 AND 7499
             UNION ALL
             SELECT 4, '2,500 - 5,000', COUNT( merkle_root)
             FROM mr_trans
             WHERE transactions BETWEEN 2500 AND 4999
             UNION ALL
             SELECT 5, '1 - 2,500', COUNT( merkle_root)
             FROM mr_trans
             WHERE transactions < 2500             
             )
             SELECT MerkleGroup, GroupCount 
             FROM mr_grouped
             ORDER BY grOrder
        """
transactions_per_merkle_root_group = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=37)

print(transactions_per_merkle_root_group)