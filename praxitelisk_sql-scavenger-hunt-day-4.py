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
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
#explore the table with head
bitcoin_blockchain.head("transactions")
# Your code goes here :)
#How many Bitcoin transactions were made each day in 2017?

query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day_of_year
            FROM time
            GROUP BY day_of_year 
            ORDER BY day_of_year
        """

print("estimated GBs needed to scanned from this query: ", bitcoin_blockchain.estimate_query_size(query))

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)
transactions_per_day.head()
import matplotlib.pyplot as plt
import seaborn as sns

# plot monthly bitcoin transactions
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("darkgrid")
sns.set(font_scale=1.2, rc={"lines.linewidth": 2.5})

plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations")
#How many transactions are associated with each merkle root?

query = """ WITH temp_table AS 
            (
                SELECT transaction_id, merkle_root
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT merkle_root, count(transaction_id) as number_of_transactions
            FROM temp_table
            GROUP BY merkle_root
            ORDER BY number_of_transactions DESC
        """

print("estimated GBs needed to scanned from this query: ", bitcoin_blockchain.estimate_query_size(query))

transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=39)
#view the result from DB with head
transactions_per_merkle_root.head()
plt.plot(transactions_per_merkle_root.number_of_transactions)
plt.title("Transactions associated with Merkle_root")