import bq_helper

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
#bitcoin_blockchain.head("blocks")
bitcoin_blockchain.head("transactions")
query_transactions1 = """    WITH transactions AS
                             (
                                 SELECT TIMESTAMP_MILLIS(timestamp) AS time,
                                        transaction_id
                                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                             )
                             SELECT EXTRACT(DAY FROM time) AS day,
                                    EXTRACT(MONTH FROM time) AS month,
                                    EXTRACT(YEAR FROM time) AS year,
                                    COUNT(transaction_id) AS n_transactions
                             FROM transactions
                             GROUP BY day, month, year
                             HAVING year = 2017
                             ORDER BY month, day
                     """
query_transactions2 = """    WITH transactions AS
                             (
                                 SELECT TIMESTAMP_MILLIS(timestamp) AS time,
                                        transaction_id
                                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                             )
                             SELECT EXTRACT(DAY FROM time) AS day,
                                    EXTRACT(MONTH FROM time) as month,
                                    COUNT(transaction_id) AS n_transactions
                             FROM transactions
                             WHERE EXTRACT(YEAR FROM time) = 2017
                             GROUP BY day, month
                             ORDER BY day, month
                     """
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_transactions1, max_gb_scanned=21)
transactions_per_day
import matplotlib.pyplot as plt
plt.bar(transactions_per_day.day, transactions_per_day.n_transactions)
plt.xlim(0, 366)
plt.xlabel("Day of the year")
plt.ylabel("Number of Transactions")
plt.title("Bitcoin Transactions in 2017")
bitcoin_blockchain.head("transactions")
query_merkle = """ WITH merkle AS
                   (
                       SELECT merkle_root AS m_root, 
                              transaction_id AS t_id
                       FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                   )
                   SELECT COUNT(t_id) AS number_transactions, m_root
                   FROM merkle
                   GROUP BY m_root
                """
trans_merkle = bitcoin_blockchain.query_to_pandas_safe(query_merkle, max_gb_scanned=37)
trans_merkle