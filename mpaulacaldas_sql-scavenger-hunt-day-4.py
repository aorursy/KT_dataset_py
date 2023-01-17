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
query_q1 = """ WITH time AS 
               (
                   SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                       transaction_id
                   FROM `bigquery-public-data.bitcoin_blockchain.transactions`
               )
               SELECT COUNT(transaction_id) AS transactions,
                   EXTRACT(DAYOFYEAR FROM trans_time) AS day_of_year,
                   EXTRACT(YEAR FROM trans_time) AS year
               FROM time
               WHERE EXTRACT(YEAR FROM trans_time) = 2017
               GROUP BY year, day_of_year 
               ORDER BY day_of_year
           """
bitcoin_blockchain.estimate_query_size(query_q1)
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_q1, max_gb_scanned=21)

transactions_per_day
bitcoin_blockchain.head(
    "transactions", 
    selected_columns = ["transaction_id", "merkle_root"]
)
query_q2_long = """
                WITH two_cols AS 
                (
                    SELECT transaction_id, merkle_root
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
                SELECT COUNT(transaction_id) AS number_of_transactions, merkle_root
                FROM two_cols
                GROUP BY merkle_root
                ORDER BY number_of_transactions DESC
                """

query_q2_short = """ SELECT COUNT(transaction_id) AS number_of_transactions, merkle_root
                     FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                     GROUP BY merkle_root
                     ORDER BY number_of_transactions DESC
                 """
bitcoin_blockchain.estimate_query_size(query_q2_long)
bitcoin_blockchain.estimate_query_size(query_q2_short)
transactions_per_root = bitcoin_blockchain.query_to_pandas_safe(query_q2_short, max_gb_scanned = 37)

transactions_per_root