# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import our bq_helper package
import bq_helper 
# create a helper object for Hacker News dataset
bitcoin = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="bitcoin_blockchain")

# print a list of all the tables in the Hacker News dataset
bitcoin.list_tables()
# check 'accident_2015' table content
bitcoin.table_schema('transactions')
bitcoin.head('transactions')
# USE CTE to get datetime timestamp from integer timestamp and then use select and group by day
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
            GROUP BY year, month, day 
            HAVING year = 2017
            ORDER BY year, month, day
        """


# check how big this query will be
bitcoin.estimate_query_size(query)
# run the query and get transactions by day
daily_transactions = bitcoin.query_to_pandas_safe(query, max_gb_scanned=21)

print(daily_transactions)
# library for plotting
import matplotlib.pyplot as plt
# make a plot
plt.plot(daily_transactions.transactions)
query = """ SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root 
            ORDER BY transactions DESC
        """
# check how big this query will be
bitcoin.estimate_query_size(query)
# run the query and get transactions by merkle
trans_per_merkle = bitcoin.query_to_pandas_safe(query, max_gb_scanned=40)
print(trans_per_merkle)
plt.plot(trans_per_merkle.transactions)