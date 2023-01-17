import pandas as pd
import bq_helper
btc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                dataset_name="bitcoin_blockchain")
btc.list_tables()
trans_dat = btc.head('transactions')
trans_dat
trans_dat.columns
#How many Bitcoin transactions were made each day in 2017?
q1 = """
WITH time AS(
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
btc.estimate_query_size(q1)
q1_ans = btc.query_to_pandas_safe(q1 ,max_gb_scanned=21)
q1_ans
trans_dat
q2 = """
    SELECT merkle_root AS merkle, COUNT(transaction_id) AS transactions
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle
    ORDER BY transactions
""" 
btc.estimate_query_size(q2)
q2_ans = btc.query_to_pandas_safe(q2 ,max_gb_scanned=37)
q2_ans