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
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )

select extract(year from trans_time) as year,
extract(month from trans_time) as month,
extract(day from trans_time) as day,
count(distinct transaction_id) as num_transactions
from time
where extract(year from trans_time) = 2017
group by 1, 2, 3
order by 1, 2, 3
"""

transactions_2017 = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
transactions_2017
query2 = """ select merkle_root, count(distinct transaction_id)
from `bigquery-public-data.bitcoin_blockchain.transactions`
group by 1
order by 2 DESC
"""

transactions_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=39)
transactions_merkle