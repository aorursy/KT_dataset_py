import bq_helper
# bigquery-public-data.bitcoin_blockchain
bbc = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                               dataset_name='bitcoin_blockchain'
                              )
bbc.list_tables()
bbc.table_schema('transactions')
bbc.head('transactions')
q1 = """
with t as (
    select
        extract(DAYOFYEAR from timestamp_millis(timestamp)) doy
    from
        `bigquery-public-data.bitcoin_blockchain.transactions`
    where
        extract(YEAR from timestamp_millis(timestamp)) = 2017
)
select
    doy,
    count(*) trans
from
    t
group by
    doy
order by
    doy
"""
bbc.estimate_query_size(q1)
trans2017 = bbc.query_to_pandas(q1)
trans2017.head()
import matplotlib.pyplot as plt

plt.plot(trans2017.trans)
plt.title("Daily Bitcoin Transactions in 2017")
plt.show()
q2 = """
select
    merkle_root
    , count(transaction_id) trans
from
    `bigquery-public-data.bitcoin_blockchain.transactions`
group by
    merkle_root
"""
bbc.estimate_query_size(q2)
trans_merkleroot = bbc.query_to_pandas(q2)
trans_merkleroot.head(20)