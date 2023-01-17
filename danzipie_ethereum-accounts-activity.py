from google.cloud import bigquery
from bq_helper import BigQueryHelper
import pandas as pd

bq_assistant = BigQueryHelper("bigquery-public-data", "ethereum_blockchain")
client = bigquery.Client()

min_block_number = 5100000
max_block_number = 6400000

# find average values and sort
query = """
SELECT
  address, SUM(n_updates) AS updates
FROM
(
  SELECT
      address, COUNT(*) AS n_updates
  FROM
  (
  SELECT DISTINCT
    from_address AS address, block_number AS block_number
  FROM
    `bigquery-public-data.ethereum_blockchain.transactions`
  WHERE
    block_number > %d
    AND
    block_number < %d
  )
  GROUP BY 
    address

  UNION ALL

  SELECT 
      address AS address, COUNT(*) AS n_updates
  FROM
  (
  SELECT DISTINCT
    to_address AS address, block_number AS block_number
  FROM
    `bigquery-public-data.ethereum_blockchain.transactions`
  WHERE
    block_number > %d
    AND
    block_number < %d
  )
  GROUP BY 
    address
)
WHERE
  n_updates >= 5
  AND
  address IS NOT NULL
GROUP BY 
  address
ORDER BY 
  updates DESC
"""

most_populars = bq_assistant.query_to_pandas_safe(query % (min_block_number, max_block_number, min_block_number, max_block_number), max_gb_scanned=40)
print("Retrieved " + str(len(most_populars)) + " accounts.")
blocks_int = max_block_number - min_block_number
most_populars = most_populars.sort_values(by='updates', ascending=False)
most_populars["probability"] = most_populars["updates"] / (blocks_int*1.0)
print(most_populars.head(10))
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c

blocks_int = max_block_number - min_block_number

# Compute probabilities
most_populars["probability"] = most_populars["updates"] / (blocks_int*1.0)
most_populars["idxs"] = range(1, len(most_populars) + 1)

# Fit curve
sol = curve_fit(func_powerlaw, most_populars["idxs"], most_populars["probability"], p0 = np.asarray([float(-1),float(10**5),0]))
fitted_func = func_powerlaw(most_populars["idxs"], sol[0][0], sol[0][1], sol[0][2])
print("Fit with values {} {} {}".format(sol[0][0], sol[0][1], sol[0][2]))

# Plot fit vs samples (only for the first 2000)
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
plt.loglog(most_populars["probability"].tolist()[1:10000],'o')
plt.loglog(fitted_func.tolist()[1:10000])
plt.xlabel("Account index (by descending popularity)")
plt.ylabel("Relative frequency [1/block]")
plt.show()
query = """
SELECT 
      timestamp, number
    FROM
      `bigquery-public-data.ethereum_blockchain.blocks`
INNER JOIN 
(
    SELECT DISTINCT
              from_address AS address, block_number AS block_number
            FROM
              `bigquery-public-data.ethereum_blockchain.transactions`
            WHERE
              from_address = '%s'
              AND
              block_number > %d
              AND
              block_number < %d

    UNION DISTINCT

    SELECT DISTINCT
              to_address AS address, block_number AS block_number
            FROM
              `bigquery-public-data.ethereum_blockchain.transactions`
            WHERE
              to_address = '%s'
              AND
              block_number > %d
              AND
              block_number < %d
) as InnerTable
ON 
    `bigquery-public-data.ethereum_blockchain.blocks`.number = InnerTable.block_number;
"""

# CryptoKitties address
adx_1 = most_populars.iloc[4].address 
transax_1 = bq_assistant.query_to_pandas_safe(query % (adx_1, min_block_number, max_block_number, adx_1, min_block_number, max_block_number), max_gb_scanned=40)
print("Retrieved " + str(len(transax_1)) + " blocks for account %s." % (adx_1) )
transax_1.sort_values(by="number", ascending=True, inplace=True)
    
# Bittrex address    
adx_2 = most_populars.iloc[3].address 
transax_2 = bq_assistant.query_to_pandas_safe(query % (adx_2, min_block_number, max_block_number, adx_2, min_block_number, max_block_number), max_gb_scanned=40)
print("Retrieved " + str(len(transax_2)) + " blocks for account %s." % (adx_2) )
transax_2.sort_values(by="number", ascending=True, inplace=True)
    
transax = list()
transax.append(transax_1)
transax.append(transax_2)
# plot the Empirical CDF
plt.figure(figsize=(15,5))
for t in transax:
    t.sort_values(by="number", inplace=True)
    tx_d = t.diff()
    tx_d = tx_d.iloc[1:]
    count = np.sort(tx_d["number"].values)
    cdf = np.arange(len(count)+1)/float(len(count))
    plt.plot(count, cdf[:-1])

plt.axis([0, 20, 0, 1])
plt.xlabel("Number of blocks without updates [n]")
plt.ylabel("Empirical CDF ( Pr[x <= n] )")
plt.show()
# activity during time
txp_1 = transax_1["timestamp"].groupby(transax_1["timestamp"].dt.floor('d')).size().reset_index(name='CryptoKitties')
txp_2 = transax_2["timestamp"].groupby(transax_2["timestamp"].dt.floor('d')).size().reset_index(name='Bittrex')
txp_1 = txp_1[10:-10]
txp_2 = txp_2[10:-10]


fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax = txp_1.plot(x="timestamp", y="CryptoKitties", ax=ax)
ax = txp_2.plot(x="timestamp", y="Bittrex", ax=ax)
plt.ylabel("Active blocks/day")

# patterns
f = plt.figure(figsize=(15,5))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)

plt.subplot(1, 2, 1)
txp_1 = transax_1["timestamp"].groupby(transax_1["timestamp"].dt.day_name()).count().sort_values()
txp_1 /= sum(txp_1)
txp_1.plot(kind="bar", ax=ax)
plt.xlabel("Day of the week")
plt.ylabel("Normalized count")
plt.title("CryptoKitties")

plt.subplot(1, 2, 2)
txp_2 = transax_2["timestamp"].groupby(transax_2["timestamp"].dt.day_name()).count().sort_values()
txp_2 /= sum(txp_2)
txp_2.plot(kind="bar", ax=ax2)
plt.xlabel("Day of the week")
plt.ylabel("Normalized count")
plt.title("Bittrex")