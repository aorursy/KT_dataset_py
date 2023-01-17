# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ -- Create Common Table 
            WITH time AS (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, 
                    transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            -- query from CTE table
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY from trans_time) AS Day,
                EXTRACT(MONTH from trans_time) AS Month,
                EXTRACT(YEAR from trans_time) AS Year            
            FROM time
            GROUP BY  Year, Month, Day
            ORDER BY transactions DESC

"""
transaction_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)
# print 
print(transaction_per_day.head(10))
bitcoin_blockchain.list_tables()
# import pyplot
import matplotlib.pyplot as plt

# plot daily transaction
plt.plot(transaction_per_day.transactions)
plt.title('Daily Bitcoin Transaction')

bitcoin_blockchain.head('transactions')
query_2 = """ WITH root AS (
                SELECT transaction_id,
                    merkle_root 
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
            SELECT COUNT(transaction_id) AS transactions,
                (merkle_root) AS m_root
            FROM root
            GROUP BY m_root
            ORDER BY m_root

"""
transaction_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query_2, max_gb_scanned=40)
print(transaction_per_merkle.head(10))
# plot transaction associated with merkle_root
plt.plot(transaction_per_merkle.transactions)
plt.title('Transaction Associated with merkle_root')
