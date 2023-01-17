import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head('transactions')
# Question 1
trayr= """With time As
            (SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
            transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`)
            SELECT Count(transaction_id) AS transactions,
            EXTRACT(YEAR FROM trans_time) AS year,
            EXTRACT (MONTH FROM trans_time) AS month,
            EXTRACT (DAY FROM trans_time)  AS day
            FROM time 
            Group by year, month, day
            HAVING year = 2017
            Order by year, month, day 
            """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(trayr, max_gb_scanned=21)
transactions_per_day.head()

import matplotlib.pyplot as plt

plt.plot(transactions_per_day.transactions)
plt.title("Transactions Per Day in 2017")

# Question 2
merk= """select merkle_root as merkle_root, count(transaction_id) as transactions
        from `bigquery-public-data.bitcoin_blockchain.transactions`
        Group by merkle_root
        Order by count(transaction_id) DESC
        """

bitcoin_blockchain.estimate_query_size(merk)

Transactions_per_Merkle_Root= bitcoin_blockchain.query_to_pandas_safe(merk, max_gb_scanned=37)

Transactions_per_Merkle_Root.head()

plt.plot(Transactions_per_Merkle_Root.transactions)
plt.title("Transactions Per Merkle Root")
