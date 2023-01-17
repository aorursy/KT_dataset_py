import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='bitcoin_blockchain')
bitcoin_blockchain.head('transactions')
query = """
WITH times AS
(
    SELECT TIMESTAMP_MILLIS(timestamp) AS timestamp, transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT
    COUNT(transaction_id) as num_transactions,
    EXTRACT(MONTH FROM timestamp) AS month,
    EXTRACT(YEAR FROM timestamp) AS year
FROM times
GROUP BY year, month
ORDER BY year, month
"""
monthly_num_transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
import matplotlib.pyplot as plt
plt.plot(monthly_num_transactions['num_transactions'])
query = """
WITH times AS
(
    SELECT TIMESTAMP_MILLIS(timestamp) AS timestamp, transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT
    COUNT(transaction_id) as num_transactions,
    EXTRACT(DAYOFYEAR FROM timestamp) AS dayofyear
FROM times
WHERE EXTRACT(YEAR FROM timestamp) = 2017
GROUP BY dayofyear
ORDER BY dayofyear
"""
daily_transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=25)
plt.plot(daily_transactions_2017.num_transactions)
plt.xlabel('Day of Year')
plt.ylabel('Number of Transactions')
plt.title('Daily Bitcoin Transactions in 2017')
plt.ylim([0,5e5])
bitcoin_blockchain.head('transactions')
query = """
SELECT
    merkle_root,
    COUNT(transaction_id) AS num_transactions
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY num_transactions
"""
root_transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=50)
# Scan size of 36 GB
root_transactions.shape
root_transactions.sort_values(by='num_transactions',ascending=False,inplace=True)
root_transactions.head(10)