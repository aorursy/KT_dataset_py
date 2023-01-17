# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head("transactions",5)
query = '''
            with time as
            (
                select TIMESTAMP_MILLIS(timestamp)as trans_time,transaction_id from `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            select count(transaction_id) as count_transid,
                    extract(day from trans_time) as days,
                    extract(month from trans_time) as months,
                    extract(year from trans_time) as year
                    from time
                    group by year,months,days
                    having year = 2017
                    order by year,months,days
        '''
bitcoin = bitcoin_blockchain.query_to_pandas_safe(query,max_gb_scanned=23)
bitcoin.head()
query2 = '''select merkle_root,count(transaction_id) as transaction from `bigquery-public-data.bitcoin_blockchain.transactions` group by merkle_root order by transaction desc
            '''
trans_per_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned = 45)
trans_per_root.head()