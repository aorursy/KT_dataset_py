# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query1 = """WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions, 
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month, day
            HAVING year = 2017
            ORDER BY month, day
         """
print("Size of query:", bitcoin_blockchain.estimate_query_size(query1))
# put data into dataframe
trans2017 = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
import matplotlib.pyplot as plt
import pandas as pd

# use pandas to transfer fetched data into dates to use for plotting
dates = pd.to_datetime(trans2017[['year', 'month', 'day']])

plt.plot(dates, trans2017.transactions)
plt.title("Transactions on each day in 2017")
query2 = """WITH data AS
            (
                SELECT merkle_root, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(DISTINCT transaction_id) as transactions, merkle_root
            FROM data
            GROUP BY merkle_root
            ORDER BY transactions DESC
         """
print("Size of query:", bitcoin_blockchain.estimate_query_size(query2))
# put into pandas dataframe
transactions_per_merkleroot = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
# show head of the dataframe
transactions_per_merkleroot.head()