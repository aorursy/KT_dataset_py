# import package with helper functions 
import bq_helper

#Import plotting library
import matplotlib.pyplot as plt

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
transact_query = """ WITH transact AS 
            (
                SELECT DATE(TIMESTAMP_MILLIS(timestamp)) AS transaction_date,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT transaction_date AS date, COUNT(transaction_id) AS transactions
            FROM transact
            WHERE EXTRACT(YEAR FROM transaction_date) = 2017
            GROUP BY date 
            ORDER BY date
        """

transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(transact_query, max_gb_scanned=21)
plt.figure(figsize=(12,6))
plt.plot(transactions_per_day_2017.date, transactions_per_day_2017.transactions)
plt.title('Bitcoin Transactions in 2017 Per Day')
plt.xlabel('Day in 2017')
plt.ylabel('Number of Transactions')
print(transactions_per_day_2017.head(10))
print()
print('Number of total entries : '+ str(len(transactions_per_day_2017)))
merkle_query = """ WITH m_root AS 
            (
                SELECT merkle_root AS Merkle,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS Transactions, merkle
            FROM m_root
            GROUP BY merkle 
            ORDER BY Transactions DESC
        """

blocks_per_merkle = bitcoin_blockchain.query_to_pandas_safe(merkle_query, max_gb_scanned=40)
top20_merkle = blocks_per_merkle.head(20)
print('Total Number of Different Merkle Roots : ' + str(len(blocks_per_merkle)))
print()
print('Top 20 Merkle Root Counts')
print(top20_merkle)