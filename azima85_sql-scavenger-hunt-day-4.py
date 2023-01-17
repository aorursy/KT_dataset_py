# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
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
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=24)
transactions_per_month
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)
query1="""WITH time AS(SELECT transaction_id AS transactions,
                                TIMESTAMP_MILLIS(timestamp) AS trans_time FROM `bigquery-public-data.bitcoin_blockchain.transactions`)
                                SELECT COUNT(transactions) AS number_of_transactions, EXTRACT(DAYOFYEAR FROM trans_time) AS day FROM time WHERE EXTRACT(YEAR FROM trans_time)=2017 GROUP BY day"""
transactions_by_day_2017 = bitcoin_blockchain.query_to_pandas(query1)
transactions_by_day_2017
plt.scatter(transactions_by_day_2017['day'],transactions_by_day_2017['number_of_transactions'])

bitcoin_blockchain.head('transactions')
query2="""SELECT merkle_root,COUNT(transaction_id) AS no_of_transactions FROM `bigquery-public-data.bitcoin_blockchain.transactions` GROUP BY merkle_root"""
transactions_by_merkle_root=bitcoin_blockchain.query_to_pandas(query2)
transactions_by_merkle_root
