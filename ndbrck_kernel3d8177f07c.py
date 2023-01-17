# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
from collections import namedtuple

FileInfo = namedtuple("FileInfo", ["is_etf", "is_stock", "symbol", "region"])



def get_file_info(path, filename):

    dirname = os.path.split(path)[1]

    if dirname == "ETFs":

        is_etf = True

    elif dirname == "Stocks":

        is_etf = False

    else:

        return None

    

    is_stock = not is_etf

    symbol, region, _ = filename.split(".")

    return FileInfo(is_etf=is_etf, is_stock=is_stock, symbol=symbol, region=region)

    



stocks_info = {}

etfs_info = {}



stocks = {}

etfs = {}

for dirname, _, filenames in os.walk('/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data'):

    for filename in filenames:

        file_info = get_file_info(dirname, filename)

        if file_info.is_etf:

            goes_to = etfs_info

            data_goes_to = etfs

        else:

            goes_to = stocks_info

            data_goes_to = stocks

            

        try:

            data_goes_to[file_info.symbol] = pd.read_csv(os.path.join(dirname, filename))

            goes_to[file_info.symbol] = file_info

        except pd.errors.EmptyDataError:

            print("Warning: Could not parse file %s" % os.path.join(dirname, filename))



# check if the symbols are also unique across stocks and etfs

if not stocks.keys().isdisjoint(etfs.keys()):

    raise RuntimeWarning("ETF and stock found sharing the same symbol: " + str(set(stocks.keys()).intersection(etfs.keys())))



info = stocks_info.copy()

info.update(etfs_info)



data = stocks.copy()

data.update(etfs)

    
# Explore the datasets

print(stocks["goog"])

print("Traded volume: %d" % np.sum(stocks["goog"]["Volume"] * stocks["goog"]["Close"]))



print(stocks["goog"][list(map(lambda s: s.startswith("2017"), stocks["goog"]["Date"]))])
num_symbols = 150



# find the most traded stocks

volume = {}

for key in stocks.keys():

    traded_recently = stocks[key][list(map(lambda s: s.startswith("2017"), stocks[key]["Date"]))]

    volume[key] = np.sum(traded_recently["Volume"] * traded_recently["Close"])



symbols = sorted(stocks.keys(), key=lambda key: volume[key], reverse=True)

symbols = symbols[:num_symbols]



print("Using the most traded %d symbols out of %d available symbols" % (len(symbols), len(stocks.keys())))



frames = []

for symbol in symbols:

    symbol_df = stocks[symbol]

    symbol_df = symbol_df[["Date", "Close"]]

    symbol_df.columns = ("Date", symbol)

    symbol_df.set_index("Date", inplace=True)

    frames.append(symbol_df)



    

print("Joining ... This may take some time")

df_joined = pd.concat(frames, axis=1, sort=False)

df_joined.sort_index(inplace=True)



df_pct = df_joined.pct_change()

df_normalized = (df_pct - df_pct.mean()) / df_pct.std()



print(df_normalized)

print("Done!")
%matplotlib inline

import matplotlib.pyplot as plt

from sklearn import preprocessing



cov_matrix = df_normalized.cov(min_periods=10)

cov_matrix = cov_matrix.abs()

cov_matrix.fillna(0, inplace=True)



plt.figure(figsize=(1.5 * len(symbols) // 2,len(symbols) // 2))

plt.imshow(cov_matrix, cmap="inferno")

plt.colorbar()



plt.gca().xaxis.set_ticks_position("both")

plt.gca().yaxis.set_ticks_position("both")

plt.gca().tick_params("both", labeltop=True, labelleft=True)



plt.xticks(range(len(symbols)), symbols)

plt.yticks(range(len(symbols)), symbols)



plt.show()
import networkx as nx

threshold_upper = 0.6

threshold_lower = 0.45



adj_matrix = cov_matrix.copy()

#adj_matrix[adj_matrix < threshold] = threshold

#adj_matrix = (adj_matrix - threshold) / (1 - threshold)

#adj_matrix **= 2

adj_matrix.clip(threshold_lower, threshold_upper, inplace=True)

adj_matrix = (adj_matrix - threshold_lower) / (threshold_upper - threshold_lower)

adj_matrix *= 4

adj_matrix **= 4



graph = nx.from_pandas_adjacency(adj_matrix)

plt.figure(figsize=(25,25))



k = 15

pos = nx.spring_layout(graph, k=k, iterations=5000, weight="weight", seed=0)

nx.draw_networkx(graph, pos, width=0.2)

plt.show()
from sklearn import cluster

kmeans = cluster.KMeans(n_clusters = len(symbols) // 5, tol=1e-2)

kmenas = cluster.DBSCAN(metric="correlation")

#kmeans = cluster.SpectralClustering(n_clusters = len(symbols) // 4, affinity="nearest_neighbors")

#kmeans = cluster.AgglomerativeClustering(n_clusters=len(symbols)//4, linkage="complete")

df_transpose = df_normalized.transpose().copy()

df_transpose.fillna(method="ffill", inplace=True, axis=1)

df_transpose.fillna(method="bfill", inplace=True, axis=1)

df_transpose = df_transpose[(column for column in df_transpose.columns if column.startswith("201"))]

#df_transpose = (df_transpose - df_transpose.mean()) / df_transpose.std()



labels = kmeans.fit_predict(df_transpose)

symbols_np = np.array(symbols)

for label in set(labels):

    print(symbols_np[labels == label])
offset = 10



df_shifted = df_normalized.filter([s for s  in df_normalized.index if s.startswith("201") ], axis=0).copy()

for symbol in symbols:

    df_shifted[symbol + "-" + str(offset)] = df_normalized[symbol].shift(-offset)



df_shifted = (df_shifted - df_shifted.mean()) / df_shifted.std()

shifted_cov = df_shifted.cov(min_periods=10)

shifted_cov = shifted_cov[[(symbol + "-" + str(offset)) for symbol in symbols]]

shifted_cov.columns = symbols

shifted_cov = shifted_cov.filter(symbols, axis=0)



shifted_cov = shifted_cov.abs()

plt.figure(figsize=(1.5 * len(symbols) // 2,len(symbols) // 2))

plt.imshow(shifted_cov, cmap="inferno")

plt.colorbar()



plt.gca().xaxis.set_ticks_position("both")

plt.gca().yaxis.set_ticks_position("both")

plt.gca().tick_params("both", labeltop=True, labelleft=True)



plt.xticks(range(len(symbols)), symbols)

plt.yticks(range(len(symbols)), symbols)



plt.show()
plt.figure(figsize=(25,25))



shifted_adj = shifted_cov.copy()

shifted_adj = shifted_adj > (shifted_adj.max().max() * 0.5)

graph = nx.from_pandas_adjacency(shifted_adj, create_using=nx.DiGraph)

k = 0.01

#pos = nx.spring_layout(graph, k=k, iterations=50, weight="weight", seed=0)

#nx.draw_networkx(graph, pos, width=0.2)

pos = nx.kamada_kawai_layout(graph)

nx.draw_networkx(graph, pos, width=0.2)

plt.show()