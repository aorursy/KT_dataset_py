# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head('transactions')
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
transactions_per_month.head()
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
import seaborn as sns

_, ax = plt.subplots(figsize=(10,7))
sns.barplot(x='month', y='transactions', data=transactions_per_month, ax=ax, ci=None)
sns.set()
sns.despine()
sns.set_style('white')
sns.set_style('ticks')

plt.title('Number of bitcoin transactions in each month')
plt.show()
# Your code goes here :)
query2 = """ WITH time AS 
                    ( SELECT  TIMESTAMP_MILLIS(timestamp) as transaction_time, transaction_id
                        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    )
                        SELECT EXTRACT(DAYOFYEAR FROM transaction_time) AS day, COUNT(transaction_id) AS n_transactions 
                        FROM time
                        WHERE EXTRACT(YEAR FROM transaction_time) = 2017
                        GROUP BY day
                        ORDER BY day
                        
                         """
transaction_year_2017 = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
transaction_year_2017.tail()

_, ax = plt.subplots(figsize=(12,9))

g = sns.pointplot(x='day',y='n_transactions',data = transaction_year_2017,ax=ax)
sns.despine()
sns.set_style('white')
sns.set_style('ticks')

plt.show()
