import numpy as np
import pandas as pd

from bq_helper import BigQueryHelper

# Helper object for BigQuery Bitcoin dataset
btc = BigQueryHelper(active_project="bigquery-public-data", 
                     dataset_name="cyrpto_bitcoin",
                     max_wait_seconds=1800)
query = """
-- get all addresses ever used as input in a transaction
WITH blockAddresses AS (
    SELECT 
      tx.block_number AS block_number
      ,tx.block_timestamp AS block_timestamp
      ,address
    FROM `bigquery-public-data.crypto_bitcoin.inputs` tx
    CROSS JOIN UNNEST(tx.addresses) AS address
),

-- generate chronological sequence of address uses
addressUses AS (
    SELECT
      block_number
      ,block_timestamp
      ,address
      ,row_number() OVER(PARTITION BY address ORDER BY block_number ASC) AS instance
    FROM blockAddresses
)

-- isolate the first block a given address was used in (instance = 1)
SELECT block_number, block_timestamp, count(*) AS new_addresses
FROM addressUses
WHERE instance = 1
GROUP BY block_number, block_timestamp
ORDER BY block_number ASC
"""

# Estimate how big this query will be
btc.estimate_query_size(query)
%%time

# Store the results into a Pandas DataFrame
df = btc.query_to_pandas_safe(query, max_gb_scanned=65)
df.head()
from matplotlib.ticker import FuncFormatter

df_tmp = df.copy()
df_tmp['dt'] = pd.to_datetime(df_tmp['block_timestamp'], unit='ms')

mean = df_tmp.set_index('dt')['new_addresses'].resample('1D').sum().rolling(30).mean()

ax = mean.plot(figsize=(16, 9), color='black')
ax.set(xlabel=None)
ax.set_title('30 day average of first-time BTC sending addresses', fontsize=20)
ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
import matplotlib.pyplot as plt

yearsBack = 6

yoy = df_tmp.set_index('dt')['new_addresses'].resample('1D').sum().rolling(30).mean().pct_change(365)[-(yearsBack*365):].dropna()

colors = yoy.map(lambda v: 'blue' if v > 0 else 'red')

axm = yoy.plot(figsize=(16, 9), color=colors, kind='bar')

axm.set(xlabel=None)
axm.set(ylabel='Year-over-year Growth [%]')
axm.set_title('30 day average of first-time sending address *growth*', fontsize=20)

divs = 60

axm.get_xaxis().set_major_formatter(plt.FixedFormatter(yoy.index.to_series().dt.strftime("%b %Y")[::divs]))
axm.locator_params(axis='x', nbins=len(yoy.index.values) / divs)

for tick in axm.get_xticklabels():
    tick.set_rotation(45)

axm.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: "{:.02f} %".format(x * 100)))
ax3 = df_tmp.set_index('dt')['new_addresses'].resample('1D').sum().cumsum().plot(figsize=(16, 9), color='red')
ax3.set(xlabel=None)
ax3.set_title('Cummulative first-time sending addresses over time', fontsize=20)
ax3.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax2 = df.plot(kind='scatter', x="block_number", y="new_addresses", figsize=(16, 9), s=1, color='purple')
ax2.set(xlabel=None)
ax2.set_title('First-time sending addresses over time (unaggregated)', fontsize=20)
ax2.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax = df.hist('new_addresses', figsize=(16, 9), bins=100)