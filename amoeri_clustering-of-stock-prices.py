import numpy as np

import pandas as pd

import plotly.express as px

import plotly.figure_factory as ff
infos = pd.read_excel('/kaggle/input/sp-500/SP500_Infos.xlsx')

infos.info()
infos.head()
stocks = pd.read_excel('/kaggle/input/sp-500/SP500_Stock_Prices.xlsx')

stocks.info()
stocks.head()
stocks.tail()
# check for empty fields

np.where(pd.isna(stocks))
# eliminate series that do not have the maximum length

stocks = stocks[stocks['Symbol'].isin(stocks['Symbol'].value_counts()[stocks['Symbol'].value_counts() == stocks['Symbol'].value_counts().max()].index.values)]
# get rid of unused colums

stocks = stocks[['Symbol', 'Date', 'Close']]
px.line(stocks, x='Date', y='Close', color='Symbol')
# normalize closing price

stocks = stocks.assign(Normalized_Close = stocks.groupby('Symbol').transform(lambda Close: (Close - Close.mean()) / Close.std()))[['Symbol', 'Date', 'Normalized_Close']]
px.line(stocks, x='Date', y='Normalized_Close', color='Symbol')
stocks_wide = stocks.pivot(index='Symbol', columns='Date', values='Normalized_Close')

from scipy.spatial.distance import squareform, pdist

euclidian_distance_matrix = pd.DataFrame(squareform(pdist(stocks_wide, metric='euclidean')), columns=stocks.Symbol.unique(), index=stocks.Symbol.unique())

px.imshow(euclidian_distance_matrix)
from scipy.cluster.hierarchy import dendrogram, average

hierarchical_euclidian_cluster = average(pdist(stocks_wide, metric='euclidean'))

fig = ff.create_dendrogram(hierarchical_euclidian_cluster)

fig.update_layout(width=800, height=500)

fig.show()
from scipy.cluster.hierarchy import cut_tree

hierarchical_euclidian_cluster_10 = pd.DataFrame(cut_tree(hierarchical_euclidian_cluster, n_clusters=10), columns=['Cluster'], index=stocks.Symbol.unique())

stocks_hec10 = stocks.join(hierarchical_euclidian_cluster_10, on='Symbol', how='left')

px.line(stocks_hec10, x='Date', y='Normalized_Close', color='Symbol', facet_col='Cluster', facet_col_wrap=4)
# for efficacy we computed the cluters once and load that result

dtw_cluster_10 = pd.read_json('../input/sp-500/dtw_cluster_10.json')

# if you'd rather wait 1-2h hours instead of 1-2s please feel free to uncomment the three lines below

# from tslearn.utils import to_time_series_dataset

# from tslearn.clustering import TimeSeriesKMeans

# dtw_cluster_10 = pd.DataFrame(TimeSeriesKMeans(n_clusters=10, metric='dtw').fit(to_time_series_dataset(stocks_wide.values)).labels_, columns=['Cluster'], index=stocks.Symbol.unique())

stocks_dtw10 = stocks.join(dtw_cluster_10, on='Symbol', how='left')

px.line(stocks_dtw10, x='Date', y='Normalized_Close', color='Symbol', facet_col='Cluster', facet_col_wrap=4)