# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

query1 = """ WITH ts AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM (TIMESTAMP_MILLIS(timestamp))) = 2017
            )
            SELECT 
                DATE(EXTRACT(YEAR FROM trans_time), EXTRACT(MONTH FROM trans_time), EXTRACT(DAY FROM trans_time)) as Date,
                COUNT(transaction_id) AS transactions
            FROM ts
            GROUP BY Date 
            ORDER BY Date
        """
transactions_per_day2017 = bitcoin_blockchain.query_to_pandas(query1)

print('How many Bitcoin transactions were made each day in 2017?')
print (transactions_per_day2017)
import matplotlib.pyplot as plt

ts = transactions_per_day2017

plt.plot(ts.Date, ts.transactions)
plt.title("Bitcoin transactions in 2017")
query2 = """ WITH mr AS 
            (
                SELECT merkle_root as MerkleRoots,
                    transaction_id as Transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT 
MerkleRoots,
COUNT(Transactions) as NbTransactions
            FROM mr
            GROUP BY MerkleRoots 
            ORDER BY NbTransactions DESC
        """
mr_and_transactions = bitcoin_blockchain.query_to_pandas(query2)

print('How many transactions are associated with each merkle root?')
print (mr_and_transactions)