# Import dataset from big query
from google.cloud import bigquery
import pandas as pd

# Connect to Google Big Query
client = bigquery.Client()

# Get 2018 average price of Ethereum
query = """
SELECT 
  SUM(value/POWER(10,18)) AS sum_tx_ether,
  AVG(gas_price*(receipt_gas_used/POWER(10,18))) AS avg_tx_gas_cost,
  DATE(timestamp) AS tx_date
FROM
  `bigquery-public-data.ethereum_blockchain.transactions` AS transactions,
  `bigquery-public-data.ethereum_blockchain.blocks` AS blocks
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

# Ethereum is volatile
#Find most popular ethereum collectables by contract count
query = """
SELECT contracts.address, COUNT(1) AS tx_count
FROM `bigquery-public-data.ethereum_blockchain.contracts` AS contracts
JOIN `bigquery-public-data.ethereum_blockchain.transactions` AS transactions ON (transactions.to_address = contracts.address)
WHERE contracts.is_erc721 = TRUE
GROUP BY contracts.address
ORDER BY tx_count DESC
LIMIT 10
"""
query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10
df.head(10)


plt.figure();
df.plot(kind='bar');
df.describe()


plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
f, g = plt.subplots(figsize=(12, 9))


plt.show( sns.lineplot(x="address", y="tx_count", data=df, palette="Blues_d"))
ax = df.hist(bins=12, xlabelsize=1.5, ylabelsize=1.5)
