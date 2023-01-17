import os
import datetime
import networkx as nx
import numpy as np
import pandas as pd

# ploty
import matplotlib.pyplot as plt 

import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()
print(os.listdir('../input'))
def plotTimeSeries(ts_txs, ts_votes):
    votes_dates = ts_votes.index
    lba = df_votes["LBA_number"].tolist()
    n_lba = df_votes["LBA_user"].tolist()
    loc = df_votes["LOC_number"].tolist()
    n_loc = df_votes["LOC_user"].tolist()
    mith = df_votes["MITH_number"].tolist()
    n_mith = df_votes["MITH_user"].tolist()
    nkn = df_votes["NKN_number"].tolist()
    n_nkn = df_votes["NKN_user"].tolist()
    poly = df_votes["POLY_number"].tolist()
    n_poly = df_votes["POLY_user"].tolist()
    
    txs_dates = ts_txs.index
    cum_amount = ts_txs["amount"].cumsum()
    
    data = [
        go.Scatter(
            x=votes_dates,
            y=lba,
            text=[f"({x[0]}, {x[1]})" for x in zip(lba,n_lba)],
            hoverinfo='name+text',
            name='lba'
        ),
        go.Scatter(
            x=votes_dates,
            y=loc,
            text=[f"({x[0]}, {x[1]})" for x in zip(loc,n_loc)],
            hoverinfo='name+text',
            name='loc'
        ),
        go.Scatter(
            x=votes_dates,
            y=nkn,
            text=[f"({x[0]}, {x[1]})" for x in zip(nkn,n_nkn)],
            hoverinfo='name+text',
            name='nkn'
        ),
        go.Scatter(
            x=votes_dates,
            y=mith,
            text=[f"({x[0]}, {x[1]})" for x in zip(mith,n_mith)],
            hoverinfo='name+text',
            name='mith'
        ),
        go.Scatter(
            x=votes_dates,
            y=poly,
            text=[f"({x[0]}, {x[1]})" for x in zip(poly,n_poly)],
            hoverinfo='name+text',
            name='poly'
        ),
        go.Scatter(
            x=txs_dates,
            y=cum_amount,
            text=[f"{x:.2f}" for x in cum_amount],
            hoverinfo='name+text',
            name='accumulate txs amount',
            yaxis='y2',
            line = dict(
                width = 4,
                dash = 'dash')
            ),
    ]
    layout=dict(
        title= "BNB Vote",
        showlegend= True,
        yaxis=dict(
            title='votes'
        ),
        yaxis2=dict(
            title='txs',
            titlefont=dict(
                color='rgb(148, 103, 189)'
            ),
            tickfont=dict(
                color='rgb(148, 103, 189)'
            ),
            overlaying='y',
            side='right'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
def plotTimeSeriesDiff(ts_txs, ts_votes):
    votes_dates = ts_votes.index
    
    lba = df_votes["LBA_number"].diff(1).fillna(0).tolist()
    n_lba = df_votes["LBA_user"].diff(1).fillna(0).tolist()
    r_lba = (df_votes["LBA_number"].diff(1).fillna(0) / df_votes["LBA_user"].diff(1).fillna(0)).fillna(0)
    
    loc = df_votes["LOC_number"].diff(1).fillna(0).tolist()
    n_loc = df_votes["LOC_user"].diff(1).fillna(0).tolist()
    r_loc= (df_votes["LOC_number"].diff(1).fillna(0) / df_votes["LOC_user"].diff(1).fillna(0)).fillna(0)
    
    mith = df_votes["MITH_number"].diff(1).fillna(0).tolist()
    n_mith = df_votes["MITH_user"].diff(1).fillna(0).tolist()
    r_mith = (df_votes["MITH_number"].diff(1).fillna(0) / df_votes["MITH_user"].diff(1).fillna(0)).fillna(0)
    
    nkn = df_votes["NKN_number"].diff(1).fillna(0).tolist()
    n_nkn = df_votes["NKN_user"].diff(1).fillna(0).tolist()
    r_nkn = (df_votes["NKN_number"].diff(1).fillna(0) / df_votes["NKN_user"].diff(1).fillna(0)).fillna(0)
    
    poly = df_votes["POLY_number"].diff(1).fillna(0).tolist()
    n_poly = df_votes["POLY_user"].diff(1).fillna(0).tolist()
    r_poly = (df_votes["POLY_number"].diff(1).fillna(0) / df_votes["POLY_user"].diff(1).fillna(0)).fillna(0)
    

    ts_txs_group = ts_txs.groupby(pd.TimeGrouper('5Min'))['amount'].apply(lambda x: sum(x) / (len(x) or 1))
    
    data = [
        go.Bar(
            x=votes_dates,
            y=r_lba,
            base=0,
            text=[f"({x[0]}, {x[1]}) {x[0] / (x[1] or 1)}" for x in zip(lba,n_lba)],
            hoverinfo='name+text',
            name='lba'
        ),
        go.Bar(
            x=votes_dates,
            y=r_loc,
            base=500,
            text=[f"({x[0]}, {x[1]}) {x[0] / (x[1] or 1)}" for x in zip(loc,n_loc)],
            hoverinfo='name+text',
            name='loc'
        ),
        go.Bar(
            x=votes_dates,
            y=r_nkn,
            base=1000,
            text=[f"({x[0]}, {x[1]}) {x[0] / (x[1] or 1)}" for x in zip(nkn,n_nkn)],
            hoverinfo='name+text',
            name='nkn'
        ),
        go.Bar(
            x=votes_dates,
            y=r_mith,
            base=1500,
            text=[f"({x[0]}, {x[1]}) {x[0] / (x[1] or 1)}" for x in zip(mith,n_mith)],
            hoverinfo='name+text',
            name='mith'
        ),
        go.Bar(
            x=votes_dates,
            y=r_poly,
            base=2000,
            text=[f"({x[0]}, {x[1]}) {x[0] / (x[1] or 1)}" for x in zip(poly,n_poly)],
            hoverinfo='name+text',
            name='poly'
        ),
        go.Bar(
            x=ts_txs_group.index,
            y=ts_txs_group,
            base=2500,
            text=[f"{x:.2f}" for x in ts_txs_group],
            hoverinfo='name+text',
            name='transactions'
        ),
    ]
    layout=dict(
        title= "5-min Increase Ratio",
        showlegend= True,
        barmode="relative",
        yaxis=dict(
            title='votes'
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
df_votes = pd.read_csv('../input/bnb_r8_votes.csv', delimiter=',')
df_votes.dataframeName = 'bnb_r8_votes.csv'
nRow, nCol = df_votes.shape
print(f'There are {nRow} rows and {nCol} columns')
end_of_votes = datetime.datetime(2018, 7, 30, 0, 6)
df_votes['created_at']  = pd.to_datetime(df_votes['created_at'])
df_votes.drop(df_votes[df_votes.created_at > end_of_votes].index, inplace=True)
df_votes = df_votes.set_index(pd.DatetimeIndex(df_votes['created_at']))
print(f'Votes Recrod Start from {min(df_votes.index)} to {max(df_votes.index)}')
df_votes.tail(5)
df_txs = pd.read_csv('../input/bnb_on_chain.csv', delimiter=',')
df_txs.dataframeName = 'bnb_on_chain.csv'
nRow, nCol = df_txs.shape
print(f'There are {nRow} rows and {nCol} columns')
df_txs = df_txs.set_index(pd.DatetimeIndex(df_txs['datetime']))
print(f'TXs Recrod was start from {min(df_txs.index)} to {max(df_txs.index)}')
print(f'Remove txs after {max(df_votes.index)}')
df_txs.drop(df_txs[df_txs.index > max(df_votes.index)].index, inplace=True)
df_txs.tail(5)
df_txs["amount"].describe().apply(lambda x: format(x, 'f'))
G = nx.DiGraph()
G.add_weighted_edges_from([(r.from_address, r.to_address, r.amount) for _,r
                           in df_txs.iterrows()])
print(nx.info(G))
def _basic_statistics(l):
    l = sorted(l)
    print(f'length : {len(l)}')
    print(f'min    : {l[0]}')
    print(f'q1     : {l[len(l)//4]}')
    print(f'q2     : {l[len(l)//2]}')
    print(f'q3     : {l[len(l)//4 * 3]}')
    print(f'max    : {l[-1]}')
    print(f'average: {sum(l)/len(l):.2f}')

in_degrees = sorted([G.in_degree(n) for n in G])
print('**All in_degrees statistics**')
_basic_statistics(in_degrees)
print()
print('**Top 50 in_degrees statistics**')
_basic_statistics(in_degrees[-50:])
remove_address_in_degrees_gte_4 = set([n for n in G if G.in_degree(n) > 4])
G.remove_nodes_from(remove_address_in_degrees_gte_4)
print(f'Removed {len(remove_address_in_degrees_gte_4)} nodes')
print(nx.info(G))

df_txs.drop(df_txs[df_txs.to_address.isin(remove_address_in_degrees_gte_4)].index, inplace=True)
df_txs.drop(df_txs[df_txs.from_address.isin(remove_address_in_degrees_gte_4)].index, inplace=True)
out_degrees = sorted([G.out_degree(n) for n in G])
print('**All out_degrees statistics**')
_basic_statistics(out_degrees)
print()
print('**Top 50 out_degrees statistics**')
_basic_statistics(out_degrees[-50:])
recv_amount = sorted([sum([e[2]['weight'] for e in G.in_edges(n, data=True)]) for n in G])
print('**All recived_amount statistics**')
_basic_statistics(recv_amount)
print()
print('**Top 50 recived_amount statistics**')
_basic_statistics(recv_amount[-50:])
recv_amount_lt_1000 = sorted([x for x in recv_amount if x < 1000])
_basic_statistics(recv_amount_lt_1000)
sns.distplot(recv_amount_lt_1000, kde=False, rug=True);
plt.show()
# maybe we left only 250 to 800
nodes_we_interested = set()
for n in G:
    s = sum([e[2]['weight'] for e in G.in_edges(n, data=True)])
    if s >= 250 and s < 800:
        nodes_we_interested.add(n)
print(len(nodes_we_interested))
df_txs.drop(df_txs[~df_txs.to_address.isin(nodes_we_interested)].index, inplace=True)

df_txs["amount"].describe().apply(lambda x: format(x, 'f'))
plotTimeSeries(df_txs, df_votes)
plotTimeSeriesDiff(df_txs, df_votes)
