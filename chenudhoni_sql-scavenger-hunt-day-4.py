# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head("transactions")
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
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
print(transactions_per_month)
####VISUALISATION:
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
#How many Bitcoin transactions were made each day in 2017?

query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            
                SELECT COUNT(transaction_id) AS no_of_transactions
                FROM time
                where EXTRACT(YEAR FROM trans_time)=2017
        """
transactions_in_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
print(transactions_in_2017)

#How many transactions are associated with each merkle root?
bitcoin_blockchain.head("transactions")
query = """ WITH t1 AS 
            (
                SELECT merkle_root,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            
                SELECT merkle_root,COUNT(transaction_id) AS no_of_transactions
                FROM t1
                group by merkle_root
                order by no_of_transactions DESC
        """
transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=40)
print("Pinting only few records")
print(transactions.head())

