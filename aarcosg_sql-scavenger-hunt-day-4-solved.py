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
# check transactions table
bitcoin_blockchain.head('transactions')
# query to find out how many Bitcoin transactions where made
# each day in 2017
query = """ WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAYOFYEAR from trans_time) AS day_of_year
            FROM time
            WHERE EXTRACT(YEAR from trans_time) = 2017
            GROUP BY day_of_year
            ORDER BY day_of_year
        """
transactions_per_day_in_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# check results
transactions_per_day_in_2017.head()
# plot results
plt.plot(transactions_per_day_in_2017.day_of_year, transactions_per_day_in_2017.transactions)
plt.title('Number of Bitcoin transactions per day in 2017')
# get the day with the most Bitcoin transactions
most_transactions_day = transactions_per_day_in_2017.loc[
    transactions_per_day_in_2017.transactions.idxmax()]
print('In 2017, the day with the most Bitcoin transactions is the {}th with {} transactions.'
      .format(most_transactions_day['day_of_year'], most_transactions_day['transactions']))
# query to find out how many transactions are associated with
# each merkle root
query = """ SELECT COUNT(transaction_id) AS transactions,
                merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions DESC
        """
transactions_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=38)
# check results
transactions_by_merkle_root.head()
print('Number of transactions associated with each merkle root')
print(transactions_by_merkle_root)
print('Merkle root {} is the one with more transactions: {}'
      .format(transactions_by_merkle_root.merkle_root[0],
              transactions_by_merkle_root.transactions[0]))