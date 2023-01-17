import bq_helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='bitcoin_blockchain')
bitcoin_blockchain.table_schema('transactions')
bitcoin_blockchain.head('transactions')
query = """WITH time AS 
(SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
FROM `bigquery-public-data.bitcoin_blockchain.transactions`)
SELECT 
COUNT(transaction_id) AS transactions,
EXTRACT(MONTH from trans_time) AS month,
EXTRACT(YEAR from trans_time) AS year
FROM time
GROUP BY year, month
ORDER BY year, month
"""
bitcoin_blockchain.estimate_query_size(query)
query = """WITH time AS
(SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
FROM `bigquery-public-data.bitcoin_blockchain.transactions`)
SELECT 
COUNT(transaction_id) AS transactions,
EXTRACT (DAY from trans_time) AS day,
EXTRACT (MONTH from trans_time) AS month,
EXTRACT (YEAR from trans_time) as year
FROM time
GROUP BY year,month,day
HAVING year = 2017
ORDER BY month,day
"""
bitcoin_blockchain.estimate_query_size(query)
transactions_per_day = bitcoin_blockchain.query_to_pandas(query)
transactions_per_day.head()
transactions_per_day.shape
plt.figure(figsize=(16,12))
plt.plot(transactions_per_day.transactions)
plt.ylabel('No. of Transactions Per Day')
plt.xlabel('Days in 2017')
bitcoin_blockchain.head('transactions')
query = """SELECT COUNT(transaction_id) AS transactions, merkle_root
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY transactions DESC
"""
bitcoin_blockchain.estimate_query_size(query)
transactions_per_merkle = bitcoin_blockchain.query_to_pandas(query)
transactions_per_merkle.head()
plt.plot(transactions_per_merkle.merkle_root,transactions_per_merkle.transactions)