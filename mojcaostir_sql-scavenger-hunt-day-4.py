import bq_helper

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                             dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head('transactions')
query = """WITH time AS
           (
               SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                   transaction_id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT COUNT(transaction_id) AS transactions,
           EXTRACT (MONTH FROM trans_time) AS month,
           EXTRACT (YEAR FROM trans_time) AS year
           FROM time
           GROUP BY year, month
           ORDER BY year, month
        """

transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
import matplotlib.pyplot as plt

plt.plot(transactions_per_month.transactions)
plt.title('Number of transaction per month')
query1= """WITH time AS
           (
               SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,transaction_id
               FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           SELECT COUNT(transaction_id) AS transactions,
               EXTRACT(DAYOFYEAR FROM trans_time) AS day,
               EXTRACT(YEAR FROM trans_time) AS year
               FROM time
               
           GROUP BY year, day
           HAVING year = 2017
           ORDER BY day
"""

bitcoin_blockchain.estimate_query_size(query1)
transactions_per_day=bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=21)
transactions_per_day
plt.plot(transactions_per_day.transactions)
query2="""SELECT COUNT(transaction_id) AS transaction,
                 merkle_root AS root
          FROM `bigquery-public-data.bitcoin_blockchain.transactions`
          GROUP BY root
          ORDER BY transaction DESC
"""

bitcoin_blockchain.estimate_query_size(query2)
merkle = bitcoin_blockchain.query_to_pandas(query2)
merkle
plt.plot(merkle.transaction)


