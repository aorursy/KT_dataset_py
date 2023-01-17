# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")


#How many Bitcoin transactions were made each day in 2017?
query1= """ WITH transa_day AS
        (
        SELECT TIMESTAMP_MILLIS(timestamp) as time1, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM time1) AS day,
                EXTRACT(MONTH FROM time1) AS month,
                EXTRACT(YEAR FROM time1) AS year
        FROM transa_day
        WHERE EXTRACT(YEAR FROM time1) = 2017
        GROUP BY year, month, day
        ORDER BY year, month, day
        """
transactions_per_day=bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)

import matplotlib.pyplot as plt

plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transactions")
#How many transactions are associated with each merkle root?

Query2= """ SELECT COUNT(transaction_id) AS No_of_transactions, merkle_root 
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY No_of_transactions DESC
        """
transactions_per_merkle=bitcoin_blockchain.query_to_pandas_safe(Query2, max_gb_scanned=40)

transactions_per_merkle.head()