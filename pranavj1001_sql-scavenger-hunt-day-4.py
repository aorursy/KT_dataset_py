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

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

# check out the table
bitcoin_blockchain.head("transactions")

query_for_first_question = """ WITH time AS 
                               (
                                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                                        transaction_id
                                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                )
                                SELECT COUNT(transaction_id) AS transactions,
                                    EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                                    EXTRACT(DATE FROM trans_time) AS date
                                FROM time
                                WHERE EXTRACT(YEAR FROM trans_time) = 2017
                                GROUP BY day, date
                                ORDER BY day, date
                           """

# check the size of the query
# bitcoin_blockchain.estimate_query_size(query_for_first_question)

# run the query
transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query_for_first_question, max_gb_scanned=21)

# print 
transactions_per_day_2017

# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day_2017.transactions)
plt.title("Daily Bitcoin Transcations in 2017")

query_for_second_question = """ WITH merkle_table AS
                               (
                                   SELECT merkle_root AS merkle_root, 
                                          transaction_id AS trans_id
                                   FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                               )
                               SELECT COUNT(trans_id) AS number_of_transactions, merkle_root
                               FROM merkle_table
                               GROUP BY merkle_root
                            """

# check the size of the query
bitcoin_blockchain.estimate_query_size(query_for_second_question)

# run the query
merkle = bitcoin_blockchain.query_to_pandas_safe(query_for_second_question, max_gb_scanned=37)

# print the result
merkle