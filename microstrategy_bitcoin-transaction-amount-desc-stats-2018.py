from google.cloud import bigquery
import pandas as pd
client = bigquery.Client()

query = """
#standardSQL
WITH
  unnested_transactions AS (
  SELECT
    T.transaction_id,
    T.block_id,
    (O.output_satoshis / 100000000) AS amount,
    DATETIME_ADD(DATETIME '1970-01-01',
      INTERVAL T.TIMESTAMP MILLISECOND) AS datetime_from_timestamp,
    CAST(DATETIME_ADD(DATETIME '1970-01-01',
        INTERVAL T.TIMESTAMP MILLISECOND) AS DATE) AS date_from_timestamp
  FROM
    `bigquery-public-data.bitcoin_blockchain.transactions` T
  CROSS JOIN
    UNNEST(outputs) O ),
  transaction_amounts AS (
  SELECT
    UT.transaction_id,
    UT.date_from_timestamp AS transaction_date,
    SUM(UT.amount) AS tot_amount
  FROM
    unnested_transactions UT
  GROUP BY
    UT.transaction_id,
    UT.date_from_timestamp )
SELECT --DISTINCT
  --AVG(TA.tot_amount) AS Mean
  PERCENTILE_CONT(TA.tot_amount,
    0.5) OVER () AS median_transaction
  ,PERCENTILE_CONT(TA.tot_amount, 0) OVER() AS min_transaction
  ,PERCENTILE_CONT(TA.tot_amount, 1) OVER() AS max_transaction
  ,PERCENTILE_CONT(TA.tot_amount, 0.25) OVER() AS quartile1
  ,PERCENTILE_CONT(TA.tot_amount, 0.75) OVER() AS quartile3
  ,STDDEV_POP(TA.tot_amount) OVER() AS standard_dev
  ,(PERCENTILE_CONT(TA.tot_amount, 0.75) OVER()) - (PERCENTILE_CONT(TA.tot_amount, 0.25) OVER()) AS IQR
  ,PERCENTILE_CONT(TA.tot_amount, 0.25) OVER() - 1.5 * (PERCENTILE_CONT(TA.tot_amount, 0.75) OVER() - PERCENTILE_CONT(TA.tot_amount, 0.25) OVER()) AS lower_limit
  ,PERCENTILE_CONT(TA.tot_amount, 0.75) OVER() + 1.5 * (PERCENTILE_CONT(TA.tot_amount, 0.75) OVER() - PERCENTILE_CONT(TA.tot_amount, 0.25) OVER()) AS upper_limit
  
FROM
  transaction_amounts TA
WHERE EXTRACT(YEAR FROM transaction_date) = 2018
LIMIT 1
"""
query_job = client.query(query)
iterator = query_job.result(timeout=1200)
rows = list(iterator)
# Transform the rows into a nice pandas dataframe
transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
# Look at the first 10 headlines
transactions.head(10)
transactions.to_csv('transactions.csv', sep='\t', encoding='utf-8')
