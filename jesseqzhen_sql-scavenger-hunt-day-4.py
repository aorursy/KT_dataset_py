# import package with helper functions 
import bq_helper
import matplotlib.pyplot as plt
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
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY month, day 
            ORDER BY month, day
        """
bitcoin_blockchain.estimate_query_size(query)
trans_by_monthday = bitcoin_blockchain.query_to_pandas(query)
trans_by_monthday
plt.plot(trans_by_monthday.transactions)
plt.title("Daily Bitcoin Transcations in 2017");
query = """
        SELECT merkle_root, COUNT(transaction_id) AS count
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        ORDER BY count DESC"""
bitcoin_blockchain.estimate_query_size(query)
merkle_root_count = bitcoin_blockchain.query_to_pandas(query)
merkle_root_count
