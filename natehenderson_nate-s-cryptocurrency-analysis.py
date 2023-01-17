import numpy as np

import pandas as pd

import glob

import matplotlib as plt

from subprocess import check_output



print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Reads in the index list "100 List.csv"

summary = pd.read_csv("../input/Top100Cryptos/100 List.csv")

cols_to_drop = ["Price","Circulating Supply", "Volume (24h)", "% Change (24h)"]

summary.drop(cols_to_drop,inplace=True, axis=1)
# Creates an array of dataframes, top_100_dfs by importing the .csv files

top_100_dfs = []

for currency in summary["Name"]:

    file_name = "../input/Top100Cryptos/" + currency + ".csv"

    df = pd.read_csv(file_name)

    top_100_dfs.append(df)
# Updates summary to include the age of each currency (on coinmarketcap.com) in the "Age" column

summary["Age"] = None

for idx, df in enumerate(top_100_dfs):

    summary["Age"][idx] = df.shape[0]
bitcoin = top_100_dfs[0]

for idx, row in enumerate(bitcoin):

    bitcoin["Daily Change"] = bitcoin["Close"] - bitcoin["Open"]

    bitcoin["Daily Change %"] = bitcoin["Daily Change"] / bitcoin["Open"]

    bitcoin["Daily Swing"] = bitcoin["High"] - bitcoin["Low"]

    bitcoin["Daily Swing %"] = bitcoin["Daily Swing"] / bitcoin["Open"]
bitcoin.max()
#graph of Bitcoin's volatility over time

BTC_volatility = bitcoin.plot(x="Date",y="Daily Swing %").invert_xaxis()
for currency in top_100_dfs:

    for idx, row in enumerate(currency):

        currency["Daily Change"] = None

        currency["Daily Change %"] = None

        currency["Daily Swing"] = None

        currency["Daily Swing %"] = None

        currency["Daily Change"] = currency["Close"] - currency["Open"]

        currency["Daily Change %"] = currency["Daily Change"] / currency["Open"]

        currency["Daily Swing"] = currency["High"] - currency["Low"]

        currency["Daily Swing %"] = currency["Daily Swing"] / currency["Open"]
bitcoin_average_daily_volatility = bitcoin["Daily Swing %"].abs().mean()

bitcoin_average_daily_volatility
# Updates summary to show average daily volatility, measured as

# as the average of the absolute value of Daily Swing %

summary["Average Daily Volatility"] = None

for idx, currency in enumerate(summary["Name"]):

    summary["Average Daily Volatility"].iloc[idx] = top_100_dfs[idx]["Daily Swing %"].abs().mean()
summary_sorted_by_volatility = summary.sort_values(by="Average Daily Volatility",ascending=False)

summary_sorted_by_volatility
# Note: Cluster dataframe doesn't include "Name"

cluster_columns = ["Market Cap", "Age", "Average Daily Volatility"]

summary_to_cluster = summary.select(lambda x: x in cluster_columns, axis=1)

summary_to_cluster["Market Cap"] = summary_to_cluster["Market Cap"].apply(lambda x: x.replace("$",""))

summary_to_cluster["Market Cap"] = summary_to_cluster["Market Cap"].apply(lambda x: x.replace(",",""))

summary_to_cluster["Market Cap"] = summary_to_cluster["Market Cap"].apply(lambda x: int(x))
# K-means cluster analysis not especially helpful

import sklearn

import sklearn.model_selection

import sklearn.cluster

sample_df = summary_to_cluster

sample_df_train, sample_df_test = sklearn.model_selection.train_test_split(sample_df, train_size=0.6, test_size=0.4)



cluster = sklearn.cluster.KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)

cluster.fit(sample_df_train)

results = cluster.predict(sample_df_test)

sample_df_test["Cluster"] = None

for idx, cluster_num in enumerate(results):

     sample_df_test["Cluster"].iloc[idx] = cluster_num

sample_df_test.sort_values(by="Cluster",inplace=True)

sample_df_test
bitcoin.head()
summary["7 Day Delta"] = None

for idx, currency in enumerate(top_100_dfs):

    if len(currency) > 7:

        start_idx = summary["Age"].iloc[idx]

        seven_day_idx = summary["Age"].iloc[idx - 7]

        seven_day_delta = currency[idx] - currency[seven_day_idx]

summary
summary



# Number 'inf' shows up where assumedly negative numbers would show

# This would make sense if these currencies were contenders for the top coin

# Ethereum, [Bitcoin Cash], Ripple, Dash, (Litecoin), NEM, Monero, [NEO], why is Factom on the list?
plt.pyplot.show(top_100_dfs[22]["Date","Market Cap"])