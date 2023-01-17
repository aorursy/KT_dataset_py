# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# import package with helper functions
import bq_helper
# create a helper object for this dataset
bitcoin = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                             dataset_name="bitcoin_blockchain")
bitcoin.list_tables()
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
# max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin.query_to_pandas_safe(query, max_gb_scanned =21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transactions")
bitcoin.head("transactions")
query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS tran_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            
            SELECT EXTRACT(DATE FROM tran_time) AS date,
                   COUNT(transaction_id) as trans_id            
            FROM time
            WHERE EXTRACT(YEAR FROM tran_time)=2017
            GROUP BY date
            ORDER BY date
             
            """
# check out data usage for the query
bitcoin.estimate_query_size(query2)
transactions_per_year = bitcoin.query_to_pandas_safe(query2, max_gb_scanned =21)
transactions_per_year.head()
# build a query
query3 = """
        SELECT COUNT(transaction_id) as transactions, merkle_root
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        ORDER BY transactions DESC
"""

# check the data usage of our query
bitcoin.estimate_query_size(query3)
trans_per_root = bitcoin.query_to_pandas_safe(query3, max_gb_scanned=37)
trans_per_root.head(10)
trans_per_root.tail(10)
