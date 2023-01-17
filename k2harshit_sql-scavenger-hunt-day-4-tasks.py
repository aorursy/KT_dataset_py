# import package with helper functions 
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
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
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
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day
            ORDER BY year, month, day
        """
bitcoin_blockchain.estimate_query_size(query)
bitcoin_trans_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
bitcoin_trans_per_day_2017.head()
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(12, 9)) 
plt.plot(bitcoin_trans_per_day_2017['transactions'])
plt.xlabel('day')
plt.ylabel('no of transactions')
plt.title("Daily Bitcoin Transcations in 2017")
f, ax = plt.subplots(figsize=(12, 9)) 
plt.scatter(bitcoin_trans_per_day_2017['day'],
            bitcoin_trans_per_day_2017['transactions'],
           c=bitcoin_trans_per_day_2017['month'])
plt.xlabel('day number')
plt.ylabel('no of transactions')
plt.title("Daily Bitcoin Transcations in 2017 colored by 'month'")
query = """SELECT COUNT(block_id) AS num_blocks,
            merkle_root
        FROM `bigquery-public-data.bitcoin_blockchain.blocks`
        GROUP BY merkle_root
        ORDER BY num_blocks DESC 
        """
bitcoin_blockchain.estimate_query_size(query)
block_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=0.1)
block_per_merkle.head()
block_per_merkle.plot()