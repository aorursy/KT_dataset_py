from google.cloud import bigquery

import pandas as pd





client = bigquery.Client()



# CALLS with NO Transfer of Value



query = """

SELECT

  COUNT(traces.transaction_hash) AS call_no_transf_tx_count,

  DATE(blocks.timestamp) AS tx_date

FROM 

  `bigquery-public-data.crypto_ethereum.traces` AS traces

JOIN `bigquery-public-data.crypto_ethereum.blocks` AS blocks

    ON traces.block_number = blocks.number

WHERE trace_type='call' 

  AND value = 0 

  AND status = 1 

  AND transaction_hash NOT IN (SELECT transaction_hash FROM `bigquery-public-data.crypto_ethereum.token_transfers`)



GROUP BY tx_date

HAVING tx_date >= '2018-01-01' AND tx_date <= '2019-12-31'

ORDER BY tx_date

"""

query_job = client.query(query)



iterator = query_job.result(timeout=200)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

df_call_no_transf = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



df_call_no_transf.head(10)
# CALLS WITH Transfer of Value



query = """

SELECT

  COUNT(traces.transaction_hash) AS call_w_transf_tx_count,

  DATE(blocks.timestamp) AS tx_date

FROM 

  `bigquery-public-data.crypto_ethereum.traces` AS traces

JOIN `bigquery-public-data.crypto_ethereum.blocks` AS blocks

    ON traces.block_number = blocks.number

WHERE (trace_type='call' 

  AND value > 0 

  AND status = 1) 

  OR (transaction_hash IN (SELECT transaction_hash FROM `bigquery-public-data.crypto_ethereum.token_transfers`))



GROUP BY tx_date

HAVING tx_date >= '2018-01-01' AND tx_date <= '2019-12-31'

ORDER BY tx_date

"""

query_job = client.query(query)



iterator = query_job.result(timeout=200)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

df_call_w_transf = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



df_call_w_transf.head()
# Average Gas price per DAY in ETH



query = """

SELECT

  AVG(transactions.gas_price/POWER(10,18)) AS avg_gas_price,

  DATE(blocks.timestamp) AS tx_date

FROM 

  `bigquery-public-data.crypto_ethereum.transactions` AS transactions

JOIN `bigquery-public-data.crypto_ethereum.blocks` AS blocks

    ON transactions.block_number = blocks.number



GROUP BY tx_date

HAVING tx_date >= '2018-01-01' AND tx_date <= '2019-12-31'

ORDER BY tx_date

"""

query_job = client.query(query)



iterator = query_job.result(timeout=200)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

df_gas_price = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



df_gas_price.head(10)
df_1 = pd.merge(df_gas_price, df_call_w_transf, on='tx_date')

df = pd.merge(df_1, df_call_no_transf, on='tx_date')

df.head(50)
df.to_csv('Analysis.csv', index=False)