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
import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = '''with time as (
                 select timestamp_millis(timestamp) as trans_time, transaction_id
                 from `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           select count(transaction_id) as transactions,
                  extract(day from trans_time) as day,
                  extract(month from trans_time) as month, 
                  extract(year from trans_time) as year
           from time
           where extract(year from trans_time) = 2017
           group by day, month, year
           order by year, month, day
        '''
bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = '''with associate as (
                 select transaction_id, merkle_root
                 from `bigquery-public-data.bitcoin_blockchain.transactions`
           )
           select merkle_root, count(transaction_id) as transactions
           from associate
           group by merkle_root
           order by merkle_root
        '''
bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)