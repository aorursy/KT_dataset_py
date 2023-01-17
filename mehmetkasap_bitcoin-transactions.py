# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head('transactions')
query1 = '''WITH time AS
            (
             SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, 
                    transaction_id
             FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
             SELECT COUNT(transaction_id) AS transactions,
                    EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                    EXTRACT(YEAR FROM trans_time) AS year
             FROM time
             GROUP BY year, day
             ORDER BY year, day
         '''
daily_transactions = bitcoin_blockchain.query_to_pandas(query1)
daily_transactions.head(10)
daily_transactions_2017 = daily_transactions[daily_transactions.year == 2017]
daily_transactions_2017.head()
import matplotlib.pyplot as plt
x = daily_transactions_2017.day
y = daily_transactions_2017.transactions
plt.figure(figsize = (12,6))
plt.plot(x,y)
plt.title('Bitcoin transactions made each day in 2017')
plt.xlabel('day')
plt.ylabel('transactions')
plt.show()
query2 = '''SELECT COUNT(transaction_id) AS Transactions, 
                   merkle_root AS Merkle_Root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY Transactions
         '''
merkle = bitcoin_blockchain.query_to_pandas(query2)
merkle