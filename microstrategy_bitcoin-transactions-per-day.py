from google.cloud import bigquery
import pandas as pd
client = bigquery.Client()

query = """
#standardSQL
SELECT * FROM (
SELECT /*EXTRACT( YEAR FROM (CAST(DATETIME_ADD(DATETIME '1970-01-01', INTERVAL T.TIMESTAMP MILLISECOND) AS DATE))) AS transaction_year
        ,*/(CAST(DATETIME_ADD(DATETIME '1970-01-01', INTERVAL T.TIMESTAMP MILLISECOND) AS DATE)) AS transaction_date
        ,COUNT(DISTINCT transaction_id) AS num_transactions
        --,COUNT(transaction_id) AS num_rows
FROM `bigquery-public-data.bitcoin_blockchain.transactions` T
GROUP BY EXTRACT( YEAR FROM (CAST(DATETIME_ADD(DATETIME '1970-01-01', INTERVAL T.TIMESTAMP MILLISECOND) AS DATE))),
      (CAST(DATETIME_ADD(DATETIME '1970-01-01', INTERVAL T.TIMESTAMP MILLISECOND) AS DATE))
)
ORDER BY --transaction_year, 
transaction_date
"""
query_job = client.query(query)
iterator = query_job.result(timeout=1200)
rows = list(iterator)
# Transform the rows into a nice pandas dataframe
transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
# Look at the first 10 headlines
transactions.head(10)
transactions.to_csv('transaction_per_day.csv', sep='\t', encoding='utf-8')
line_chart = transactions.plot(kind='line')
