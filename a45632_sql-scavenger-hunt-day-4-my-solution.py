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
# Your code goes here :)
query = """    WITH transactions AS
                             (
                                 SELECT TIMESTAMP_MILLIS(timestamp) AS time,
                                        transaction_id
                                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                             )
                             SELECT EXTRACT(DAY FROM time) AS day,
                                    EXTRACT(YEAR FROM time) as year,
                                    COUNT(transaction_id) AS n_transactions
                             FROM transactions
                             WHERE EXTRACT(YEAR FROM time) = 2017
                             GROUP BY day, year
                             ORDER BY day, year
                     """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

import matplotlib.pyplot as plt
plt.bar(transactions_per_day.day, transactions_per_day.n_transactions)
plt.xlim(0, 31)
plt.xlabel("Day of the year")
plt.ylabel("Number of Transactions")
plt.title("Bitcoin Transactions in 2017")

#Part 2
query = """ WITH merkle AS
                   (
                       SELECT merkle_root AS m_root, 
                              transaction_id AS t_id
                       FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                   )
                   SELECT COUNT(t_id) AS number_transactions, m_root
                   FROM merkle
                   GROUP BY m_root
                """
transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
transactions