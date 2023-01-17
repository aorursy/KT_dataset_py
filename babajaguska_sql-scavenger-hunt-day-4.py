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
qEachDay="""WITH time AS
                (SELECT transaction_id,
                EXTRACT (DAY FROM TIMESTAMP_MILLIS(timestamp)) as day,
                EXTRACT (MONTH FROM TIMESTAMP_MILLIS(timestamp)) as month,
                EXTRACT (YEAR FROM TIMESTAMP_MILLIS(timestamp)) as year
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
SELECT month,day,COUNT(transaction_id) as trans_per_day
FROM time
WHERE year=2017
GROUP BY month,day
ORDER BY month,day
"""

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(qEachDay, max_gb_scanned=21)
print(transactions_per_day.head())

qMerkle="""SELECT merkle_root,COUNT(transaction_id) AS num_trans
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY num_trans DESC
"""

xx=bitcoin_blockchain.query_to_pandas_safe(qMerkle,max_gb_scanned=37)
print(xx.head())