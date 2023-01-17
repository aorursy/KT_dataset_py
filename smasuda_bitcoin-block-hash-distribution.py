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
#convert hexdecimal blockhash into int. Apply log scale for plot.
import math
df['blockhash_int'] = df['block_id'].apply(lambda x : int(x,16))
df['blockhash_log10'] = df['blockhash_int'].apply(lambda x : math.log10(x))

# remove the outliers
df = df[ (df['timestamp'] > 1230854400000) & (df['timestamp'] < 1600000000000) ]
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')
from matplotlib import pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 10

plt.hist(df['blockhash_log10'], bins=100, histtype='stepfilled')
plt.title('histogram of log10(blockhash)')
plt.xlabel('log10(blockhash)')
plt.ylabel('frequency')
plt.show();
plt.plot_date( df['timestamp'], df['blockhash_log10'], ms=0.1);
plt.title("log10(blockhash) vs block time");
plt.xlabel("block time");
plt.ylabel("blockhash in log10");
fig = plt.figure(figsize=(15,10))
sc=plt.scatter( df['nonce'], df['blockhash_log10'], s=0.01, c=df['timestamp']);
fig.colorbar(sc)
plt.title("log10(blockhash) vs nonce, color=block timestamp");
plt.xlabel("nonce");
plt.ylabel("blockhash in log10");
plt.show();