# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import bq_helper
bitcoin = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="bitcoin_blockchain")
bitcoin.head("blocks")
bitcoin.head("transactions")
query_0 = """WITH time AS
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

transactions_per_month = bitcoin.query_to_pandas_safe(query_0, max_gb_scanned=21)
import matplotlib.pyplot as plt

plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transactions")
query_1 = """WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, day
            HAVING year = 2017
            ORDER BY day
            """
bitcoin.estimate_query_size(query_1)
transactions_per_day = bitcoin.query_to_pandas_safe(query_1, max_gb_scanned=25)
x = transactions_per_day.day
y = transactions_per_day.transactions
plt.plot(x, y)
plt.title("Bitcoin Transactions by Day of Month")

query_2 = """SELECT merkle_root, COUNT(transaction_id) AS trans
             FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             GROUP BY merkle_root
             ORDER BY trans DESC
            """

# checking size of query
bitcoin.estimate_query_size(query_2)
merkle = bitcoin.query_to_pandas_safe(query_2, max_gb_scanned=40)

# top 10 merkle roots with most transactions
merkle.head(10)
# getting the flow of transactions through the entire 2017
query_3 = """WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DATE FROM trans_time) AS date
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY date
            ORDER BY date
            """
bitcoin.estimate_query_size(query_3)
transactions_in_2017 = bitcoin.query_to_pandas_safe(query_3, max_gb_scanned=25)
x = transactions_in_2017.date
y = transactions_in_2017.transactions
plt.plot(x, y)
plt.title("Transactions by Day in 2017")
