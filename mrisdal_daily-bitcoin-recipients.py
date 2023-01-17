from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

# Query by Allen Day, GooglCloud Developer Advocate (https://medium.com/@allenday)
query = """
#StandardSQL
SELECT o.day, COUNT(DISTINCT(o.output_key)) AS recipients FROM
(
SELECT TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day, outputs.pubkey_base58 AS output_key, satoshis 
FROM 
  `bitcoin-bigquery.bitcoin.blocks` 
    JOIN
  UNNEST(transactions) AS transactions
    JOIN 
  UNNEST(transactions.outputs) AS outputs
) AS o
GROUP BY day
ORDER BY day
"""

query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
transactions.head(10)
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline

transactions.plot()