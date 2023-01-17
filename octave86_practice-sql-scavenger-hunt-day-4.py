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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.table_schema("blocks")
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS days,
                EXTRACT(MONTH FROM trans_time) AS month
            FROM time 
            WHERE  EXTRACT(YEAR FROM trans_time) =2017
            GROUP BY month,days 
            ORDER BY month,days
        """

# note that max_gb_scanned is set to 21, rather than 1
bitcoin_blockchain.estimate_query_size(query1)
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day_2017)
plt.title("Daily Bitcoin Transcations")
transactions_per_day_2017.head(50)
query2 = """SELECT COUNT(transaction_id) AS transactions_count,
                merkle_root AS merkel_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
            GROUP BY merkle_root 
            ORDER BY transactions_count"""

bitcoin_blockchain.estimate_query_size(query2)

query3 = """SELECT COUNT(merkle_root) AS transactions_count,
                merkle_root AS merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions` 
            GROUP BY merkle_root 
            ORDER BY transactions_count"""

bitcoin_blockchain.estimate_query_size(query3)

transactions_per_merkelroot = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
transactions_per_merkelroot.head(10)


# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_merkelroot.transactions_count)
plt.title("Merkle Root Bitcoin Transcations")