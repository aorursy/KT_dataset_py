import pandas as pd

from google.cloud import bigquery



client = bigquery.Client()



amount = 0.12345678  # Amount in BTC

query = """

 SELECT

    outputs.*

 FROM

    `bigquery-public-data.crypto_bitcoin.outputs` outputs

 WHERE

    outputs.value = {}

"""



query_job = client.query(query.format(int(amount * 10**8)))



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

headlines = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

headlines.head(100)