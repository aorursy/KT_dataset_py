import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query ="""
            with time as
            (
            select transaction_id, TIMESTAMP_MILLIS(timestamp) AS trans_time
            from `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            Select EXTRACT(DAY from trans_time) as day,EXTRACT(YEAR from trans_time) as year,
            count(transaction_id) as transactions from time
            group by year, day
            having year = 2017
            order by year, day
            """
bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("daily Bitcoin Transcations")
query2 = """ with merkel_data as
            ( select transaction_id, merkle_root
            from `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            select count(transaction_id),merkle_root
            from merkel_data group by merkle_root"""
bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)