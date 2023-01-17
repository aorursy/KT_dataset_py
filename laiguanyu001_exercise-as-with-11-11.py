# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# Your Code Here
#bitcoin_blockchain.list_tables()
#bitcoin_blockchain.head("transactions")
query = """with transaction as 
(
    select timestamp_millis(timestamp) as trans_time,
        transaction_id
        from `bigquery-public-data.bitcoin_blockchain.transactions`   
)
,transaction2 as
(
    select count(transaction_id) as trans,
        extract (day from trans_time) as day,
        extract (month from trans_time) as month,
        extract (year from trans_time) as year
        from transaction
        group by day,month,year
        order by day,month,year desc
)
select trans,month,day
from transaction2
where year = 2017
"""
output = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=25)
import matplotlib.pyplot as plt
plt.plot(output.trans)
plt.title("daily Bitcoin Transcations in 2017")


# Your Code Here
bitcoin_blockchain.head("transactions")
query = """with merkle as
    ( 
        select count(transaction_id) as trans,
            merkle_root
            from `bigquery-public-data.bitcoin_blockchain.transactions`
            group by merkle_root
    )
select trans, merkle_root
    from merkle
    order by trans desc
"""
output = bitcoin_blockchain.query_to_pandas(query)
output.head()