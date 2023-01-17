# imports
import bq_helper
import matplotlib.pyplot as plt

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ WITH time AS 
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
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
#print(transactions_per_day_2017)

# plot monthly bitcoin transactions
plt.plot(transactions_per_day_2017.transactions)
plt.title("Daily Bitcoin Transactions 2017")

bitcoin_blockchain.head("transactions")
query = """
        SELECT COUNT(transaction_id) AS transactions, merkle_root AS root
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY root
        ORDER BY transactions DESC
        """
transactions_roots = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=38)
print(transactions_roots)