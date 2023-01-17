from google.cloud import bigquery

import pandas as pd



client = bigquery.Client()



# Query by Allen Day, GooglCloud Developer Advocate (https://medium.com/@allenday)

query = """

#standardSQL

SELECT

  o.day,

  COUNT(DISTINCT(o.output_key)) AS recipients

FROM (

  SELECT

    TIMESTAMP_MILLIS((timestamp - MOD(timestamp,

          86400000))) AS day,

    output.output_pubkey_base58 AS output_key

  FROM

    `bigquery-public-data.bitcoin_blockchain.transactions`,

    UNNEST(outputs) AS output ) AS o

GROUP BY

  day

ORDER BY

  day

"""



query_job = client.query(query)



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the first 10 headlines

transactions.head(10)
transactions = transactions.set_index('day')
import matplotlib

from matplotlib import pyplot as plt

%matplotlib inline



transactions.plot()
# https://github.com/SohierDane/BigQuery_Helper

from bq_helper import BigQueryHelper



# This establishes an authenticated session and prepares a reference to the dataset that lives in BigQuery.

bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")
df = bq_assistant.query_to_pandas_safe(query)
df = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=35)
print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))
df.set_index('day').plot(figsize=(30, 8))