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
query1 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT 
                COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year,
                EXTRACT(DAY FROM trans_time) AS day
            FROM time
            GROUP BY year,month,day
            HAVING year = 2017
            ORDER BY year,month,day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)

print(transactions_per_day.head(10))
print('\n# of Distinct Transaction Counts %s' % len(transactions_per_day['transactions']))

plt.figure(figsize=(14,6))
plt.hist(transactions_per_day['transactions'],bins = 50)
plt.xlabel('Transactions Per Day')
plt.ylabel('Frequency')
plt.title('Distribution of Transactions Per Day')
plt.show()
query2 = """
            SELECT COUNT(transaction_id) AS transaction_count
                , merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`       
            GROUP BY merkle_root
        """

# note that max_gb_scanned is set to 21, rather than 1
merkle_transaction = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)
print(merkle_transaction.head(10))
print('\n# of Unique Transaction Counts %s' % len(merkle_transaction['transaction_count'].unique()))
print('# of Unique Merkle Roots %s' % len(merkle_transaction['merkle_root'].unique()))
print('\nMany merkle roots had the same number of transactions.')

plt.figure(figsize=(14,6))
plt.hist(merkle_transaction['transaction_count'],bins = 100)
plt.xlabel('Transactions Per Merkle Root')
plt.ylabel('Frequency')
plt.title('Distribution of Transaction Counts Per Merkle Root')
plt.show()