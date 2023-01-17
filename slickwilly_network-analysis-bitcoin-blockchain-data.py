# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
import networkx as nx

# import plotting library
#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = [12.0, 8.0]

# display all outputs within each Juypter cell
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

# Create helper object for the  the bigQuery data set
blockchain_helper = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                            dataset_name="bitcoin_blockchain")
# inspect the structure
blockchain_helper.list_tables()
# look at a table of the information for both data sets


blockchain_helper.head('transactions')
blockchain_helper.head('blocks')
# lets parse the timestamp data into readable date times
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    block_id
                FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            )
            SELECT COUNT(block_id) AS blocks,
                EXTRACT(DATE FROM trans_time) AS date
            FROM time
            GROUP BY date
            ORDER BY date
        """
blockchain_helper.estimate_query_size(query)
q1_df = blockchain_helper.query_to_pandas(query)
plt.plot(q1_df['date'] ,q1_df['blocks'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Blocks', fontsize=18)
plt.title('Block Creation Volume Per Day', fontsize=22)
plt.bar(q1_df['date'], q1_df['blocks'], align='edge')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Blocks', fontsize=18)
plt.suptitle('Blocks Created Per Day', fontsize=22)
# when did this outlier occur?
# it looks like there was a large influx of users July 2010:
# https://en.bitcoin.it/wiki/2010#July
q1_df.sort_values('blocks', ascending=False).head(10)
# lets find which addresses has the most number of transactions
QUERY = """
    SELECT
        inputs.input_pubkey_base58 AS input_key, count(*)
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    JOIN UNNEST (inputs) AS inputs
    WHERE inputs.input_pubkey_base58 IS NOT NULL
    GROUP BY inputs.input_pubkey_base58 order by count(*) desc limit 1000
    """
blockchain_helper.estimate_query_size(QUERY)
q2 = blockchain_helper.query_to_pandas(QUERY)
q2.head(n=10)
# lets find which addresses has the most number of transactions
QUERY = """
    SELECT
        outputs.output_pubkey_base58 AS output_key, count(*)
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    JOIN UNNEST (outputs) AS outputs
    WHERE outputs.output_pubkey_base58 IS NOT NULL
    GROUP BY outputs.output_pubkey_base58 order by count(*) desc limit 1000
    """
blockchain_helper.estimate_query_size(QUERY)
q2 = blockchain_helper.query_to_pandas(QUERY)
q2.head(n=50)
# lets query all transactions this person was involved in
q_input = """
        WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    inputs.input_pubkey_base58 AS input_key,
                    outputs.output_pubkey_base58 AS output_key,
                    outputs.output_satoshis AS satoshis,
                    transaction_id AS trans_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    JOIN UNNEST (inputs) AS inputs
                    JOIN UNNEST (outputs) AS outputs
                WHERE inputs.input_pubkey_base58 = '1NxaBCFQwejSZbQfWcYNwgqML5wWoE3rK4'
                    OR outputs.output_pubkey_base58 = '1NxaBCFQwejSZbQfWcYNwgqML5wWoE3rK4'
            )
        SELECT input_key, output_key, satoshis, trans_id,
            EXTRACT(DATE FROM trans_time) AS date
        FROM time
        --ORDER BY date
        """
blockchain_helper.estimate_query_size(q_input)
q3 = blockchain_helper.query_to_pandas(q_input)
q3.head(10)

# make a datatime type transformation
q3['date'] = pd.to_datetime(q3.date)
q3 = q3.sort_values('date')
# convert satoshis to bitcoin
q3['bitcoin'] = q3['satoshis'].apply(lambda x: float(x/100000000))
print(q3.info())

# make any sending of bitcoin a negative value representing 'leaving' this wallet
q3['bitcoin_mod'] = q3['bitcoin']
q3.loc[q3['input_key'] == '1NxaBCFQwejSZbQfWcYNwgqML5wWoE3rK4', 'bitcoin_mod'] = -q3['bitcoin_mod']
# sanity check...
q3.loc[q3['input_key'] == '1NxaBCFQwejSZbQfWcYNwgqML5wWoE3rK4'].head()
q3.info()
# how many unique addresses are included in this wallets transaction history?
q3['output_key'].nunique()
# how many transactions to the top 10 addresses?
# extreme value for the top address...
q3['output_key'].value_counts().nlargest(10)
# fold difference between the largest and second largest wallet transactions - 44X!
# what is the story behind this wallet...?
q3['output_key'].value_counts().nlargest(5).iloc[0] / q3['output_key'].value_counts().nlargest(5).iloc[1]
# how many unique addresses are included in this wallets transaction history?
q3['input_key'].nunique()
# how many transactions to the top 10 addresses?
# extreme value for the top address...
q3['input_key'].value_counts().nlargest(10)
# fold difference between the largest and second largest wallet transactions - 44X!
# what is the story behind this wallet...?
q3['input_key'].value_counts().nlargest(5).iloc[0] / q3['input_key'].value_counts().nlargest(5).iloc[1]
# we should look at transaction activity across time
q3_plot = q3['date'].value_counts()
q3_plot.head()
# plotting params
ax = plt.gca()
ax.scatter(q3_plot.index, q3_plot.values)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Transactions', fontsize=18)
plt.suptitle('Volume of Transactions Per Day', fontsize=22)
#ax.set_yscale('log')
# daily difference in input transactions and output transactions
daily_diff = q3.groupby(['date'])['input_key', 'output_key'].nunique()
daily_diff['difference'] = daily_diff.input_key - daily_diff.output_key
plt.scatter(daily_diff.index, daily_diff.difference, c=daily_diff.difference, cmap='viridis')
# percentage of unique transactions out of total - the rest must be transactions to multiple addresses (don't really understand this...)
q3['trans_id'].nunique() / len(q3)
# there are multiple records of the same input to outout address recoreded with the same transaction ID
# why does this happen? is the total transferred bitcoin for a transaction being split up across several records?
len(q3.groupby(['trans_id', 'output_key']).nunique()) / len(q3)
# lets group each individual transaction and the total amount of bitcoin sent / received and plot over time
trans_plot = q3.groupby(['date', 'trans_id'])['bitcoin'].sum()
trans_plot.head(10)
# lets plot the value of transactions over time from this wallet
q4_plot = q3.groupby('date', as_index=False)[['bitcoin_mod']].sum()
plt.plot_date(q4_plot['date'], q4_plot['bitcoin_mod'])
plt.ylabel('Bitcoin', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.suptitle('Net Bitcoin Transacted Per Day', fontsize=22)
# lets get the total amount of bitcoin this wallet sent out each month
value_plot = q3.groupby([q3['date'].dt.month]).sum().reset_index()
value_plot = q3.set_index(pd.DatetimeIndex(q3['date'])).groupby(pd.TimeGrouper('M'))[['bitcoin_mod']].sum()
value_plot.head()
plt.scatter(value_plot.index, value_plot.values)
plt.xlabel('Date by Month', fontsize=18)
plt.ylabel('Bitcoin', fontsize=18)
plt.suptitle('Net Bitcoin Transacted per Month', fontsize=22)
# lets get the total amount of transactions this wallet made each month
corr_plot = q3.groupby([q3['date'].dt.month]).sum().reset_index()
corr_plot = q3.set_index(pd.DatetimeIndex(q3['date'])).groupby(pd.TimeGrouper('M'))[['trans_id']].count()
corr_plot = corr_plot.join(value_plot)
corr_plot.head()
plt.scatter(corr_plot.index, corr_plot['trans_id'])
plt.scatter(corr_plot.index, corr_plot['bitcoin_mod'])
plt.legend(fontsize=14, labels=('Volume Transactions', 'Net Bitcoin Exchange'))
plt.xlabel('Date by Month', fontsize=18)
#plt.ylabel('Transactions', fontsize=18)
plt.title('Monthly Transaction Volume & Net Bitcoin Exchange', fontsize=22)
# look at a correlation matrix between transactions in a given month
# and net bitcoins transacted in a given month
fin_corr = pd.concat([value_plot, corr_plot], axis = 1)
fin_corr.corr()

# top 20 wallets bitcoins sent out to
nx_plot_asc = q3.groupby(['input_key', 'output_key'], as_index=False)['bitcoin_mod'].sum().sort_values(by='bitcoin_mod', ascending=True)[0:20]
# top 20 wallets bitcoins received from
nx_plot_desc = q3.groupby(['input_key', 'output_key'], as_index=False)['bitcoin_mod'].sum().sort_values(by='bitcoin_mod', ascending=False)[0:20]
nx_plot = pd.concat([nx_plot_asc, nx_plot_desc], ignore_index=True)

# networkx graph of net volumne of bitcoins transacted (most into and most out of the wallet in question)
G = nx.from_pandas_edgelist(nx_plot, 'input_key', 'output_key', 'bitcoin_mod', nx.DiGraph())
# specify node color
color_list = []
for node in G.nodes():
    if node == '1NxaBCFQwejSZbQfWcYNwgqML5wWoE3rK4':
        color_list.append('red')
    else:
        color_list.append('green')
# plot
pos=nx.random_layout(G)
nx.draw_networkx_nodes(G,pos, node_list= G.nodes(),node_color=color_list,node_size=100)
nx.draw_networkx_edges(G, pos)
#nx.draw_networkx_labels(G, pos, font_size=8)
plt.title('Net Volume Bitcoins Exchanged', fontsize=22)
nx_plot_asc[0:5]
# we can query for the top x wallets that transact with our wallet of interest
# from nx_plot_asc[0:10]
# leave out the original wallet
# where do top outputs send their bitcoin?
top_out =  """
       WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    inputs.input_pubkey_base58 AS input_key,
                    outputs.output_pubkey_base58 AS output_key,
                    outputs.output_satoshis AS satoshis,
                    transaction_id AS trans_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    JOIN UNNEST (inputs) AS inputs
                    JOIN UNNEST (outputs) AS outputs
                WHERE inputs.input_pubkey_base58 = '1EBhP9kzJjZhCLDqepZa1jeg8RFrZTkd2X'
                    OR outputs.output_pubkey_base58 = '1EBhP9kzJjZhCLDqepZa1jeg8RFrZTkd2X'
                    OR inputs.input_pubkey_base58 = '3QHYuqj9EZ9FzVwpEagQwo2SQG7MhHGrw9'
                    OR outputs.output_pubkey_base58 = '3QHYuqj9EZ9FzVwpEagQwo2SQG7MhHGrw9'
                    OR inputs.input_pubkey_base58 = '1BT4DYrt3ZoSz6WeGEZzrj4tdidUcpCfQ6'
                    OR outputs.output_pubkey_base58 = '1BT4DYrt3ZoSz6WeGEZzrj4tdidUcpCfQ6'
                    OR inputs.input_pubkey_base58 = '1HdCtn5aiySHpyrRs5FSiS1NPSm9BPnEeR'
                    OR outputs.output_pubkey_base58 = '1HdCtn5aiySHpyrRs5FSiS1NPSm9BPnEeR'
                    OR inputs.input_pubkey_base58 = '1LyJffQE3iCzRdv1Fqv7Y5wawr7u6aajw8'
                    OR outputs.output_pubkey_base58 = '1LyJffQE3iCzRdv1Fqv7Y5wawr7u6aajw8'
            )
        SELECT input_key, output_key, satoshis, trans_id,
            EXTRACT(DATE FROM trans_time) AS date
        FROM time
        --ORDER BY date
        """
blockchain_helper.estimate_query_size(top_out)
# top nodes currency sent to
top_out_df = blockchain_helper.query_to_pandas(top_out)
top_out_df.head(10)

# make a datatime type transformation
top_out_df['date'] = pd.to_datetime(top_out_df.date)
top_out_df = top_out_df.sort_values('date')
# convert satoshis to bitcoin
top_out_df['bitcoin'] = top_out_df['satoshis'].apply(lambda x: float(x/100000000))

# make any sending of bitcoin a negative value representing 'leaving' this wallet
top_out_df['bitcoin_mod'] = top_out_df['bitcoin']

top_out_df.loc[(top_out_df['input_key'] == '1EBhP9kzJjZhCLDqepZa1jeg8RFrZTkd2X')
               | (top_out_df['input_key'] == '3QHYuqj9EZ9FzVwpEagQwo2SQG7MhHGrw9')
               | (top_out_df['input_key'] == '1BT4DYrt3ZoSz6WeGEZzrj4tdidUcpCfQ6')
               | (top_out_df['input_key'] == '1HdCtn5aiySHpyrRs5FSiS1NPSm9BPnEeR')
               | (top_out_df['input_key'] == '1LyJffQE3iCzRdv1Fqv7Y5wawr7u6aajw8'), 'bitcoin_mod'] = -top_out_df['bitcoin_mod']
    
# sanity check...
top_out_df.loc[top_out_df['input_key'] == '1EBhP9kzJjZhCLDqepZa1jeg8RFrZTkd2X'].head()
top_out_df.loc[top_out_df['output_key'] == '1EBhP9kzJjZhCLDqepZa1jeg8RFrZTkd2X'].head()
# visualize network of transaction volume out of our wallet in question
nx_plot = pd.Series.to_frame(top_out_df.groupby(['input_key', 'output_key'], as_index=True)['bitcoin_mod'].nunique().sort_values(ascending=False)).reset_index(['input_key','output_key'])
# create graph
G = nx.from_pandas_edgelist(nx_plot[0:200], 'input_key', 'output_key', 'bitcoin_mod', nx.MultiDiGraph())
# specify node color
color_list = []
for node in G.nodes():
    if node == '1NxaBCFQwejSZbQfWcYNwgqML5wWoE3rK4':
        color_list.append('red')
    else:
        color_list.append('green')
# draw graph
pos = nx.spring_layout(G)
#plt.figure(3,figsize=(16,12))
nx.draw_networkx_nodes(G,pos,node_color=color_list,node_size=20)
#nx.draw_networkx_labels(G, pos, font_size=4)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.suptitle('Bitcoin Exchange Network', fontsize=22, y=0.945)
plt.title('of the top 5 wallets to receive funds from 1NxaBCFQwejSZbQfWcYNwgqML5wWoE3rK4', fontsize=13)
# compute degree centrality
pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index').sort_values(by = [0], ascending=False)[0:10]
in_cent = pd.DataFrame.from_dict(nx.in_degree_centrality(G), orient='index').sort_values(by = [0], ascending=False)[0:10]
out_cent = pd.DataFrame.from_dict(nx.out_degree_centrality(G), orient='index').sort_values(by = [0], ascending=False)[0:10]
bw_cent = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index').sort_values(by = [0], ascending=False)[0:10]
cent = pd.merge(in_cent, out_cent, left_index=True, right_index=True)
cent = pd.merge(cent, bw_cent, left_index=True, right_index=True)
cent.columns = ['Indegree', 'Outdegree', 'Betweenness']
import seaborn as sns
sns.heatmap(cent, cmap='Blues', annot=True)
plt.title('Node Centrality Metrics', fontsize=22, y=1.04, x=.35)
nx_plot_desc[0:5]