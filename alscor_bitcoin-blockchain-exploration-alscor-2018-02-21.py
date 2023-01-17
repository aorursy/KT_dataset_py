import pandas as pd
from bq_helper import BigQueryHelper
bq_assist = BigQueryHelper('bigquery-public-data', 'bitcoin_blockchain')
bq_assist.table_schema(table_name='transactions')
## Get an intersect between input and output keys with highest # of txns in 2010-2016
q2 = """
#standardSQL

WITH 
sq_i1k as 
(select 
    input_pubkey_base58 as user_key, 
    count(distinct transaction_id) as n_inp_txns
 from `bigquery-public-data.bitcoin_blockchain.transactions`
 cross join unnest(inputs)
 where input_pubkey_base58 is not NULL
 and input_pubkey_base58 <> ''
 and timestamp > 1262304000000   /* 2010-01-01 */
 and timestamp < 1483228800000   /* 2017-01-01 */
 group by input_pubkey_base58
 order by n_inp_txns DESC
 limit 1000
 )
select * from  
(select 
    output_pubkey_base58 as user_key, 
    count(distinct transaction_id) as n_out_txns, 
    sum(output_satoshis/100000000) as tot_btc
 from `bigquery-public-data.bitcoin_blockchain.transactions`
 cross join unnest(outputs)
 where output_pubkey_base58 in (select distinct user_key from sq_i1k)
 and output_satoshis > 0
 and timestamp > 1262304000000   /* 2010-01-01 */
 and timestamp < 1483228800000   /* 2017-01-01 */
 group by output_pubkey_base58
 order by n_out_txns DESC
 limit 1000
 ) t
"""
res1 = bq_assist.query_to_pandas_safe(q2, max_gb_scanned=100)
print('Most active exchange (participate in both input and output transactions, 2010B-2016E')
print(res1)
pd.options.display.max_rows = 999
res1['f_lucky'] = res1.user_key.apply(lambda x: 1 if 'Lucky' in x else 0)
res1['f_dice'] = res1.user_key.apply(lambda x: 1 if 'dice' in x else 0)
res1[(res1['f_lucky']==1)|(res1['f_dice']==1)].sort_values(by='user_key')
res1.to_csv('bitcoin_most_active_io_exchanges2010_2016.csv',index=False)