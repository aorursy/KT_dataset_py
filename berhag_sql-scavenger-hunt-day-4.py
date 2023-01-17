import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import bq_helper
bitcoin = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                 dataset_name = "bitcoin_blockchain")
bitcoin.list_tables()
bitcoin.table_schema("transactions")
bitcoin.head('transactions', num_rows = 5)
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month 
        """
bitcoin.estimate_query_size(query)
df = bitcoin.query_to_pandas_safe(query, max_gb_scanned=21)
df.head(10)
# library for plotting
import matplotlib.pyplot as plt
plt.plot(df.transactions)
plt.title("Monthly transactions")
query = """ WITH time AS
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
                    GROUP BY day, month, year
                    HAVING year = 2017
                    ORDER BY month, day
                    """
bitcoin.estimate_query_size(query)
df = bitcoin.query_to_pandas_safe(query, max_gb_scanned=21)
df.head(10)
plt.plot(df.transactions)
plt.title("Daily number of bitcoin transactions")
query = """ SELECT merkle_root, COUNT(transaction_id) AS n_transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY n_transactions DESC
            """
bitcoin.estimate_query_size(query)
df = bitcoin.query_to_pandas_safe(query, max_gb_scanned= 38)
df.head(10)
plt.plot(df.n_transactions)
plt.title("number of  transactions associated with each merkle root")