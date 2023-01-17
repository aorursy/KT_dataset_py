# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

query = """
SELECT 
  timestamp,
  number,
  size,
  gas_limit,
  gas_used
FROM
  `bigquery-public-data.ethereum_blockchain.blocks` AS blocks
WHERE TRUE
  AND timestamp >= '2016-01-01 00:00:00'
  AND timestamp < '2018-01-01 00:00:00'
"""
# This establishes an authenticated session and prepares a reference to the dataset that lives in BigQuery.
bq_assistant = BigQueryHelper("bigquery-public-data", "ethereum_blockchain")
df = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=15)
f, g = plt.subplots(figsize=(12, 9))
g = sns.lineplot(x="timestamp", y="gas_used", data=df, palette="Blues_d")
plt.title("Gas used per block over time")
plt.show(g)
f, g = plt.subplots(figsize=(12, 9))
g = sns.lineplot(x="timestamp", y="gas_limit", data=df, palette="Blues_d")
plt.title("Gas Limit per block over time")
plt.show(g)
f, g = plt.subplots(figsize=(12, 9))
g = sns.lineplot(x="timestamp", y="size", data=df, palette="Blues_d")
plt.title("Size (bytes) per block over time")
plt.show(g)