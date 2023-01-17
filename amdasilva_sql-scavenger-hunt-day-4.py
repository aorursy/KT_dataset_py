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
query1 =  """ WITH time AS 
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
transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
transactions_2017.head()
plt.plot(transactions_2017.day, transactions_2017.transactions/1000.)
plt.title("Daily Bitcoin Transcations in 2017")
plt.ylabel('Number of transactions (x1,000)')
plt.xlabel('Day of the year')
bitcoin_blockchain.table_schema('transactions')
#bitcoin_blockchain.head('blocks')
query2 = """SELECT merkle_root,
                   COUNT(transaction_id) AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions DESC
         """
transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)
transactions_per_merkle_root.head()
plt.plot(transactions_per_merkle_root.transactions)
plt.title('Number of transactions in different blocks')
p = transactions_per_merkle_root['transactions'].hist(bins=20)
p.set_xlabel('Number of transactions')
p.set_ylabel('Number of blocks')
