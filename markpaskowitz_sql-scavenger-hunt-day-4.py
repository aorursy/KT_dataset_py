# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """
    WITH TransTimes AS (
        SELECT TIMESTAMP_MILLIS(timestamp) as transaction_time,
            transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
    SELECT EXTRACT(MONTH FROM transaction_time) AS month,
        EXTRACT(DAY FROM transaction_time) AS day,
        COUNT(transaction_id) AS num_transactions
    FROM TransTimes
    WHERE EXTRACT(YEAR from transaction_time) = 2017
    GROUP BY month, day
    ORDER BY month, day
    """
trans_by_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 25)
trans_by_day.head()
import matplotlib.pyplot as plt
plt.plot(trans_by_day.num_transactions)
query = """
    SELECT merkle_root,
        COUNT(transaction_id) AS num_transactions
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle_root
    ORDER BY num_transactions DESC
    """
trans_by_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 40)
plt.plot(trans_by_root.num_transactions)
trans_by_root.head()