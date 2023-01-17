# Your code goes here :)

# Import package with helper functions 
import bq_helper
import matplotlib.pyplot as plt

# Create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

# Question1: How many Bitcoin transactions were made each day in 2017?

question1 = """ WITH day AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS Transactions,
            EXTRACT(DAY FROM trans_time) AS Day, 
            EXTRACT(YEAR FROM trans_time) AS Year
            FROM day
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, day
            ORDER BY day 
        """

# Estimation of query question1
print(bitcoin_blockchain.estimate_query_size(question1))

# I use max_db_scanned = 21 to limit at 21 GB as Rachel suggest
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(question1, max_gb_scanned=21)

# Print Dataframe Size
print('Dataframe Size: {} Bytes'.format(int(transactions_per_day.memory_usage(index=True, deep=True).sum())))

# Print Dataframe "transactions_per_day"
print(transactions_per_day.head(20))
# Plot Transactions Day by Day

transactions_per_day.plot.barh(x='Day',y='Transactions',figsize=(10,8), legend=False)
plt.title('Day by Day Transacton in the 2017')
plt.ylabel('Day')
plt.xlabel('Transactions')
plt.yticks(fontsize=14)
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()
# Question2: How many transactions are associated with each merkle root?

question2 = """ SELECT merkle_root AS Merkle_Root, COUNT(transaction_id) AS Transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
            """

# Estimation of query question2
print(bitcoin_blockchain.estimate_query_size(question2))

# I use max_db_scanned = 40 because this is a huge DataFrame
merkle_root = bitcoin_blockchain.query_to_pandas_safe(question2, max_gb_scanned=40)

# Print Dataframe Size
print('Dataframe Size: {} Bytes'.format(int(merkle_root.memory_usage(index=True, deep=True).sum())))

# Print Dataframe "merkle_root"
print(merkle_root.head(20))
# Plot "merkle_root" DataFrame

f, ax = plt.subplots(figsize=(12, 9)) 
plt.plot(merkle_root['Transactions'])
plt.xlabel('Merkle_root')
plt.ylabel('Transactions')
plt.title("Merkle Root")
plt.show()