import numpy as np

import pandas as pd

import os

from google.cloud import bigquery
client = bigquery.Client()

ethereum_classic_dataset_ref = client.dataset('crypto_ethereum_classic', project='bigquery-public-data')
query = """

SELECT from_address AS address

FROM `bigquery-public-data.crypto_ethereum_classic.transactions`

WHERE block_number > 7500000

GROUP BY from_address

"""



query_job = client.query(query)

iterator = query_job.result()
rows = list(iterator)

# Transform the rows into a nice pandas dataframe

active_users = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
# Post-process: convert to airdrop-tool format

active_users['tokens'] = ['1000000000000000000' for x in rows] # to airdrop 1 token (with 18 decimals) to each address

active_users.shape
out = active_users.to_json(orient='records')

with open('airdrop.json', 'w') as f:

    f.write(out)