# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # BigQuery
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='bitcoin_blockchain')
bitcoin_blockchain.head('transactions')
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day
                
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day 
            ORDER BY day
        """

# check size of query
bitcoin_blockchain.estimate_query_size(query)
txn_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
txn_per_day
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
txn_per_day.plot(kind='line', x='day', y='transactions')
plt.show()
query = """
SELECT COUNT('block_id') AS blocks, merkle_root
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY blocks DESC
"""

# check how big this query will be
bitcoin_blockchain.estimate_query_size(query)
blocks_per_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=20)
blocks_per_root