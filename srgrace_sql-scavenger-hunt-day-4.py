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
fig = plt.figure(figsize=(12,5))

plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# Your code goes here :)

query1 = """
        WITH time AS (
            SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id) AS transactions, EXTRACT(DATE from trans_time) as date
        FROM time
        WHERE EXTRACT(YEAR from trans_time) = 2017
        GROUP BY date
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)

transactions_per_day.head()
transactions_per_day.to_csv('transactions_per_day.csv')
import seaborn as sns

ax = plt.subplots(figsize=(12, 7))
sns.set_style("whitegrid") 
ax = sns.barplot(x="date", y="transactions", data=transactions_per_day, palette="coolwarm")
plt.title("Bitcoin transactions made each day in 2017")
# OR

ax = plt.subplots(figsize=(15,7))
plt.bar(transactions_per_day.date, transactions_per_day.transactions)
plt.title("Bitcoin transactions made each day in 2017")

bitcoin_blockchain.head("transactions")
bitcoin_blockchain.head('blocks')
query2 = """
        SELECT COUNT(transaction_id) as transactions, merkle_root
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        ORDER BY transactions DESC
        """
# note that max_gb_scanned is set to 37, rather than 1
transactions_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=37)

transactions_merkle_root.head()
transactions_merkle_root.to_csv('transactions_merkle_root.csv')
