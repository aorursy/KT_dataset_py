import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
import matplotlib.pyplot as plt
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                             dataset_name="bitcoin_blockchain")
bitcoin_blockchain.table_schema("transactions")
bitcoin_blockchain.head("transactions")
query = """WITH time AS
           (
                SELECT transaction_id,
                       TIMESTAMP_MILLIS(timestamp) AS trans_time
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT EXTRACT(MONTH FROM trans_time) AS month,
                  EXTRACT(YEAR FROM trans_time) AS year,
                  COUNT(transaction_id) AS transactions
           FROM time
           GROUP BY year, month
           ORDER BY year, month
        """
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transactions")
scav1 = """WITH time_proper AS
           (
               SELECT TIMESTAMP_MILLIS(timestamp) AS date_times,
                      transaction_id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           
           SELECT DATE(date_times) AS just_dates,
                  COUNT(transaction_id) AS transactions
           FROM time_proper
           GROUP BY just_dates
           HAVING EXTRACT(YEAR FROM just_dates) = 2017
           ORDER BY just_dates
        """
daily_transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(scav1, max_gb_scanned=22)
plt.plot(daily_transactions_2017.just_dates, daily_transactions_2017.transactions)
plt.title("Daily Bitcoin Transactions in 2017")
scav2 = """SELECT merkle_root,
                  COUNT(transaction_id)
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY merkle_root
        """
merk_transacts = bitcoin_blockchain.query_to_pandas_safe(scav2, max_gb_scanned=38)
plt.plot(merk_transacts.f0_)
plt.title("Range of Transactions by Merkle Root")