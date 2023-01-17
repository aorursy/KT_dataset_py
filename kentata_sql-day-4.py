# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import bq_helper

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head("transactions")
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
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set();
transactions_per_month.head()
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transactions")
# Question1
# How many Bitcoin transactions were made each day in 2017?

# 1. convert integers to timestamp and name the CTE as time_1
# 2. extract days(1-365) and years from the timestamp from time_1 and name the CTE as time_2
# 3. count the number of transactions in 2017 ordered by day(0-365)
query_1 = """ WITH time_1 AS 
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, 
                    transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                 
             ),
             
             time_2 AS 
             (
                 SELECT transaction_id,
                 EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                 EXTRACT(YEAR FROM trans_time) AS year
                 FROM time_1
             )
            SELECT COUNT(transaction_id) AS transactions, day
            FROM time_2
            WHERE year = 2017
            GROUP BY day
            ORDER BY day
          """
transactions_2017_per_day = bitcoin_blockchain.query_to_pandas_safe(query_1, max_gb_scanned=21)
transactions_2017_per_day.head()
plt.plot(transactions_2017_per_day.transactions)
transactions_2017_per_day.head()
# Question2: How many transactions are associated with each merkle root?
bitcoin_blockchain.head("transactions")
query_2 = """ WITH time_1 AS 
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, 
                    transaction_id, merkle_root
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                 
             )
             
        
            SELECT merkle_root, COUNT(transaction_id) AS transactions
            FROM time_1
            GROUP BY merkle_root
            ORDER BY transactions DESC
          """
transactions_per_markleroot = bitcoin_blockchain.query_to_pandas_safe(query_2, max_gb_scanned=37)
transactions_per_markleroot.head()
transactions_per_markleroot.shape
