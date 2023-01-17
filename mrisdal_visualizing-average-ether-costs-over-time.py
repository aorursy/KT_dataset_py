from google.cloud import bigquery

import pandas as pd



client = bigquery.Client()



# Query by Allen Day, GooglCloud Developer Advocate (https://medium.com/@allenday)

query = """

SELECT 

  SUM(value/POWER(10,18)) AS sum_tx_ether,

  AVG(gas_price*(receipt_gas_used/POWER(10,18))) AS avg_tx_gas_cost,

  DATE(timestamp) AS tx_date

FROM

  `bigquery-public-data.crypto_ethereum.transactions` AS transactions,

  `bigquery-public-data.crypto_ethereum.blocks` AS blocks

WHERE TRUE

  AND transactions.block_number = blocks.number

  AND receipt_status = 1

  AND value > 0

GROUP BY tx_date

HAVING tx_date >= '2018-01-01' AND tx_date <= '2018-12-31'

ORDER BY tx_date

"""
query_job = client.query(query)



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the first 10

df.head(10)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})



f, g = plt.subplots(figsize=(12, 9))

g = sns.lineplot(x="tx_date", y="avg_tx_gas_cost", data=df, palette="Blues_d")

plt.title("Average Ether transaction cost over time")

plt.show(g)
# https://github.com/SohierDane/BigQuery_Helper

from bq_helper import BigQueryHelper



# This establishes an authenticated session and prepares a reference to the dataset that lives in BigQuery.

bq_assistant = BigQueryHelper("bigquery-public-data", "crypto_ethereum")
df = bq_assistant.query_to_pandas_safe(query)
df = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=18)
print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))
f, g = plt.subplots(figsize=(12, 9))

g = sns.lineplot(x="tx_date", y="avg_tx_gas_cost", data=df, palette="Blues_d")

plt.title("Average Ether transaction cost over time")

plt.show(g)