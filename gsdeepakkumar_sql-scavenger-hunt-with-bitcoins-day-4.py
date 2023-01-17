# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
import matplotlib.pyplot as plt






bitcoin = bq_helper.BigQueryHelper(active_project='bigquery-public-data',dataset_name='bitcoin_blockchain')
bitcoin.table_schema('blocks')
bitcoin.table_schema('transactions')
query = """WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,EXTRACT(DATE FROM trans_time) AS DATE FROM time
            GROUP BY DATE 
            ORDER BY DATE """
bitcoin.estimate_query_size(query)
date_trend_transactions = bitcoin.query_to_pandas_safe(query,max_gb_scanned=21)
date_trend_transactions.head()
plt.plot(date_trend_transactions.transactions)
plt.title("Transaction trend -Day wise")
query=""" WITH merkle as (SELECT merkle_root, transaction_id as transaction from `bigquery-public-data.bitcoin_blockchain.transactions`) SELECT count(transaction) as transaction_count , merkle_root as merkle from merkle group by merkle """
bitcoin.estimate_query_size(query)
bitcoin.query_to_pandas_safe(query,max_gb_scanned=50)