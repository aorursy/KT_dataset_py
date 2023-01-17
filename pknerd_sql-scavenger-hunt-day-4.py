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
print(transactions_per_month.head())
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day_year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day_year, month 
            ORDER BY day_year, month
        """
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

#plot the data
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
merkle_query = """ WITH merkle AS
                   (
                       SELECT merkle_root AS m_root, 
                              transaction_id AS t_id
                       FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                   )
                   SELECT COUNT(t_id) AS number_transactions, m_root
                   FROM merkle
                   GROUP BY m_root
                """
transactions_merkle = bitcoin_blockchain.query_to_pandas_safe(merkle_query, max_gb_scanned=37)
print(transactions_merkle)