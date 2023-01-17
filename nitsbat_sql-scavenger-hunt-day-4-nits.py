import bq_helper
bitc = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                             dataset_name = "bitcoin_blockchain")
bitc.list_tables()
bitc.table_schema('blocks')
bitc.head('blocks')
query = """ WITH time AS
            ( 
                select TIMESTAMP_MILLIS(timestamp) as trans_time,transaction_id
                from `bigquery-public-data.bitcoin_blockchain.transactions`
            )
        select count(transaction_id) as transactions,
        extract(MONTH from trans_time) as trans_month,
        extract(YEAR from trans_time) as trans_year from
        time
        group by trans_year,trans_month
        order by trans_year,trans_month
        """
bitc.estimate_query_size(query)
transactions_per_month = bitc.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_month.head()
import matplotlib.pyplot as plt
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transactions")
bitc.list_tables()
bitc.head('transactions')
bitc.table_schema('transactions')
query1 = """ WITH time as
            (
                    select TIMESTAMP_MILLIS(timestamp) as trans_time,
                    transaction_id as id
                    from `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            select count(id) as cid,
            EXTRACT(DAY from trans_time) as trans from time
            where EXTRACT(YEAR from trans_time) = 2017 
            group by trans
            order by trans
            """
bitc.estimate_query_size(query1)
res = bitc.query_to_pandas_safe(query1,max_gb_scanned = 21)
res
import matplotlib.pyplot as pal
pal.plot(res.cid)
pal.title('2017 Bitcoin Transaction (Day-wise)')
bitc.table_schema('transactions')
bitc.head('transactions')
query2 = """ select merkle_root as root,count(transaction_id) as total_id from
                `bigquery-public-data.bitcoin_blockchain.transactions`
                group by root
                order by total_id
          """
bitc.estimate_query_size(query2)
result = bitc.query_to_pandas_safe(query2,max_gb_scanned=37)
result.head()
query3 = """ select merkle_root as root,count(transaction_id) as total_id from
                `bigquery-public-data.bitcoin_blockchain.transactions`
                group by root
                order by root
          """
bitc.estimate_query_size(query3)
result = bitc.query_to_pandas_safe(query3,max_gb_scanned=37)
result.head()
