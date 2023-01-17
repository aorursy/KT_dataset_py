

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))


import bq_helper 

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="biquery-public-data",
                                            dataset_name="bitcoin_blockchain")
query = """WITH time AS
          (
           SELECT TIMESTAMP_MILLIS(timestamp) as trans_time,
                 transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT COUNT(transaction_id) AS transactions,
                  EXTRACT(MONTH FROM trans_time) AS month,
                  EXTRACT (YEAR FROM trans_time) AS year
                  
         FROM time
         GROUP BY year, month
         ORDER BY year, month
        
        """
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=25)

transactions_per_month.head()
import matplotlib.pyplot as plt

plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transactions")
query_17 = """WITH time AS
             (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
             SELECT COUNT(transaction_id) AS transactions,
                   EXTRACT(DAY FROM trans_time) AS day,
                   EXTRACT(YEAR FROM trans_time) AS year
           FROM time
           
           GROUP BY day, year
           HAVING year = 2017
           ORDER BY day, year
           
           """
daily_2017_transactions =  bitcoin_blockchain.query_to_pandas_safe(query_17, max_gb_scanned=24)
daily_2017_transactions.head()
daily_2017_transactions.describe()
plt.plot(daily_2017_transactions)
plt.title("Daily Bitcoin Transactions in 2017")
query_merkle = """WITH merkle as
                (
                  SELECT merkle_root, transaction_id
                  FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(transaction_id),
                 merkle_root 
                FROM merkle
                GROUP BY merkle_root
              """
query_merkle = bitcoin_blockchain.query_to_pandas_safe(query_merkle,max_gb_scanned=42)
query_merkle.head()
query_merkle.info()
query_merkle['merkle_root'].unique()