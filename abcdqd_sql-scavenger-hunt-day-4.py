import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bq
bc = bq.BigQueryHelper(active_project= 'bigquery-public-data',
                      dataset_name= 'bitcoin_blockchain')

# check field characteristics
bc.table_schema('transactions')
# query 1

query = """
        select
            extract (date from timestamp_millis(timestamp)) as date,
            count(transaction_id) as count
        from `bigquery-public-data.bitcoin_blockchain.transactions`
        where extract(year from timestamp_millis(timestamp)) = 2017
        group by date
        order by date
        """
bc.query_to_pandas(query)
# query 2

query = """
        select
            merkle_root,
            count(transaction_id) as transactions
        from `bigquery-public-data.bitcoin_blockchain.transactions`
        group by merkle_root
        order by transactions desc
        """
bc.query_to_pandas(query)