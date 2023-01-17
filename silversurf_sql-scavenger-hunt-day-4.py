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
q_transactions_per_day = '''
    WITH transactions_timestamped AS (
        SELECT
            TIMESTAMP_MILLIS(timestamp) AS trans_time,
            transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
    
    SELECT
        EXTRACT(YEAR FROM trans_time) AS year,
        EXTRACT(MONTH FROM trans_time) AS month,
        EXTRACT(DAY FROM trans_time) AS day,
        COUNT(transaction_id) AS transactions
    FROM transactions_timestamped
    GROUP BY year, month, day
    HAVING year=2017
    ORDER BY month, day
'''
bitcoin_blockchain.estimate_query_size(q_transactions_per_day)
transactions_daily_2017 = bitcoin_blockchain.query_to_pandas(q_transactions_per_day)
transactions_daily_2017.head()
import seaborn as sns
sns.set()
transactions_daily_2017.plot(y='transactions', title='Daily bitcoin transactions, 2017')
q_transactions_per_merkle = '''
    SELECT
        merkle_root,
        COUNT(transaction_id) AS transactions
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle_root
'''
bitcoin_blockchain.estimate_query_size(q_transactions_per_merkle)
transactions_per_merkle_df = bitcoin_blockchain.query_to_pandas(q_transactions_per_merkle)
print('merkle_roots found: ',len(transactions_per_merkle_df))
%matplotlib inline
plt.figure(figsize=(10,8))

ax = transactions_per_merkle_df.transactions.hist(bins=400, linewidth = 0.4, edgecolor='white');
ax.set_title('Transactions per merkle root distribution', alpha=0.9, size=14)
ax.set_xlim(-100, 4000);
ax.set_ylabel('Frequency', alpha = 0.8);
ax.set_xlabel('Transactions', alpha = 0.8);
plt.figure(figsize=(10,8))

ax = transactions_per_merkle_df.transactions.hist(bins=400, linewidth = 0.4, edgecolor='white');
ax.set_title('Transactions per merkle root distribution', alpha=0.9, size=14)
ax.set_xlim(50, 4000);
ax.set_ylim(0, 25000);
ax.set_ylabel('Frequency', alpha = 0.8);
ax.set_xlabel('Transactions', alpha = 0.8);