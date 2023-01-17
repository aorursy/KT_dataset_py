from google.cloud import bigquery
import pandas as pd
client = bigquery.Client()
query = "select timestamp as timestamp, nonce,block_id from `bigquery-public-data.bitcoin_blockchain.blocks`"

query_job = client.query(query)
df = query_job.to_dataframe()
#check the first 5 result, to ensure we get what we want.
#additionally do data sanity check by looking at block exploerers such as 
#https://blockexplorer.com/block/{block_id}

pd.set_option('display.max_colwidth', -1)
df.head(5)
# remove the outliers
df = df[ (df['timestamp'] > 1230854400000) & (df['timestamp'] < 1600000000000) ]
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')
from matplotlib import pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.plot_date( df['timestamp'], df['nonce'], ms=0.1);
plt.title("btc");
plt.xlabel("block timestamp");
plt.ylabel("nonce");