import bq_helper
# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
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

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
fig = plt.figure(figsize=(12,5))
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
q_day = """
    with time as (
        SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
    select count(transaction_id) AS transactions, EXTRACT(DATE from trans_time) as date
    from time 
    where EXTRACT(YEAR FROM trans_time)=2017
    group by date;
"""
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(q_day, max_gb_scanned=21)
transactions_per_day.head()
# plot monthly bitcoin transactions
# fig = plt.figure(figsize=(12,5))
ax = plt.subplots(figsize=(15,7))
plt.bar(transactions_per_day.date, transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
q_mr = """
    select count(transaction_id) as trxn_count, merkle_root
    from `bigquery-public-data.bitcoin_blockchain.transactions`
    group by merkle_root
"""

transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(q_mr, max_gb_scanned=38)
transactions_per_merkle.head()