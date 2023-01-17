import pandas as pd

from bq_helper import BigQueryHelper

pd.options.display.max_colwidth = 100

# This establishes an authenticated session and prepares a reference to the dataset that lives in BigQuery.

bq_assistant = BigQueryHelper("bigquery-public-data", "crypto_bitcoin")
query = """

select count(*)

FROM `bigquery-public-data.crypto_bitcoin.transactions`

where  DATE(block_timestamp) > '2018-01-15' and DATE(block_timestamp) > '2018-01-18' and input_value > 1111111111111

-- ORDER BY input_value DESC 



"""

df = bq_assistant.query_to_pandas_safe(query)

df
df