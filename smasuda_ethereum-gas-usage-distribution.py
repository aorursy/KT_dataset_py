from google.cloud import bigquery

import pandas as pd

client = bigquery.Client()

query = """

select format_timestamp('%Y-%m-%d' ,TIMESTAMP_TRUNC(block_timestamp, WEEK) ) as week,

round(log10(receipt_gas_used), 1) as log10_gas_used, count(*) as ntx 

from `bigquery-public-data.crypto_ethereum.transactions`

group by week, log10_gas_used

order by week

"""
query_job = client.query(query)

df = query_job.to_dataframe()

df_pivot = pd.pivot_table(data=df, values='ntx', columns='week', index='log10_gas_used')

df_pivot.head()
df_pivot = pd.pivot_table(data=df, values='ntx', columns='week', index='log10_gas_used')

df_pivot.head()
from matplotlib import pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

import numpy as np

import seaborn as sns

fig = plt.figure(figsize=(12,6))

ax=sns.heatmap(df_pivot,cmap='Reds', cbar_kws={'label': '# of txns'});

ax.invert_yaxis()
df['log10_ntx'] = np.log10( df['ntx'])

df_pivot = pd.pivot_table(data=df, values='log10_ntx', columns='week', index='log10_gas_used')



fig = plt.figure(figsize=(12,6))

ax=sns.heatmap(df_pivot,cmap='Blues', cbar_kws={'label': 'Log10(# of txns)'});

ax.invert_yaxis()
query_wk_ntx = """

select format_timestamp('%Y-%m-%d' ,TIMESTAMP_TRUNC(block_timestamp, WEEK) ) as week,

count(*) as ntx_wk 

from `bigquery-public-data.crypto_ethereum.transactions`

group by week

order by week

"""

query_job_wk_ntx = client.query(query_wk_ntx)

df_wk_ntx = query_job_wk_ntx.to_dataframe()
df_wk_ntx.head()
df.head()
df_merged = pd.merge(df, df_wk_ntx)



df_merged['ntx_ratio'] = df_merged['ntx']/df_merged['ntx_wk']

df_merged.head(30)
df_merged_pivot = pd.pivot_table(data=df_merged, values='ntx_ratio', columns='week', index='log10_gas_used')

df_merged_pivot.head()
fig = plt.figure(figsize=(15,6))

ax=sns.heatmap(df_merged_pivot,cmap='Oranges', cbar_kws={'label': 'ratio of txns'});

ax.invert_yaxis()