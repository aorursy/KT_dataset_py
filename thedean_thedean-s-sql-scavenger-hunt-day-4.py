# Import Package and helper functions

import bq_helper

#create a helper object

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                             dataset_name="bitcoin_blockchain")

# Question 1: How many BTC transactions were made in each day

query = """
        WITH time as
        (
            SELECT DATE(TIMESTAMP_MILLIS(timestamp)) AS correctTime,
                transaction_id AS transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        
        SELECT correctTime AS Day, COUNT(transactions) AS DailyTransactions
        FROM time
        WHERE EXTRACT(YEAR FROM correctTime) = 2017
        GROUP BY Day
        ORDER BY Day
        """
# Make the query for Question 1 safe

BTC_Per_Day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

# How did we do?

BTC_Per_Day.head()

# Import matplotlib

import matplotlib.pyplot as plt

# Plot the information we have gathered

plt.plot(BTC_Per_Day.DailyTransactions)
plt.title("Daily Bitcoin Transactions in 2017")

# Query 2: How many transactions are associated with each Merkle Root

query2 = """
            SELECT merkle_root, COUNT(transaction_id) as num_trans
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY num_trans DESC
            """
# To pass this code, we should first check its size    
bitcoin_blockchain.estimate_query_size(query2)

# At roughly 37GB, let's start there

trans_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned = 37)

print(trans_per_merkle)

# This may take a minute or two, so hang tight