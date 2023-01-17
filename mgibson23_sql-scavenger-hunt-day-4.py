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
# Your code goes here :)
#How many Bitcoin transactions were made each day in 2017?
#You can use the "timestamp" column from the "transactions" table to answer this question. 
#You can check the notebook from Day 3 for more information on timestamps.

query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DATE FROM trans_time) AS date
            FROM time
            GROUP BY date
            ORDER BY date
        """
    
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=21)
 
import matplotlib.pyplot as plt
print(transactions_per_day)

# plot daily bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
#How many transactions are associated with each merkle root?
#You can use the "merkle_root" and "transaction_id" columns in the "transactions" table 
#to answer this question. Note that the earlier version of this question asked 
#"How many blocks are associated with each merkle root?", which would be one block for each root.

bitcoin_blockchain.head("transactions", 
    selected_columns = ["merkle_root", "transaction_id"],
    num_rows = 10
)


query3 = """ WITH root AS 
            (
                SELECT merkle_root, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions, merkle_root
            FROM root
            GROUP BY merkle_root
            ORDER BY transactions DESC
        """

# note that max_gb_scanned is set to 37, rather than 1
transactions_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query3, max_gb_scanned=37)
print(transactions_merkle_root)
# plot daily bitcoin transactions
plt.plot(transactions_merkle_root.transactions)
plt.title("Bitcoin Transcations by Merkle Root")