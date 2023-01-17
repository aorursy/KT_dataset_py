# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head('transactions')
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
bitcoint_daily_query = """WITH LastYearData AS (
                                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                                        transaction_id 
                                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                )
                          SELECT COUNT(transaction_id) AS num_transactions,
                                 EXTRACT(DATE FROM trans_time) AS day
                          FROM LastYearData
                          WHERE trans_time < '2018-01-01'
                                AND trans_time >= '2017-01-01'
                          GROUP BY day
                          ORDER BY day
                        """
bitcoin_blockchain.estimate_query_size(bitcoint_daily_query)
daily_bitcoin = bitcoin_blockchain.query_to_pandas_safe(bitcoint_daily_query, max_gb_scanned=21)
daily_bitcoin.head()
fig = plt.figure(figsize=(12,5))
plt.plot(daily_bitcoin.day, daily_bitcoin.num_transactions)
merkleroot_query = """WITH merkleroot AS(
                            SELECT transaction_id,
                                    merkle_root
                            FROM `bigquery-public-data.bitcoin_blockchain.transactions`)
                      SELECT merkle_root,
                             COUNT(transaction_id) AS num_transactions
                      FROM merkleroot
                      GROUP BY merkle_root
                      ORDER BY num_transactions DESC
                    """

bitcoin_blockchain.estimate_query_size(merkleroot_query)
merkleroot_trans = bitcoin_blockchain.query_to_pandas_safe(merkleroot_query, max_gb_scanned=37)
merkleroot_trans.head()
