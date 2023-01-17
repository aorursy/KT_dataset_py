import numpy as np

import pandas as pd

import glob

import seaborn as sns

import networkx as nx



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.concat([pd.read_csv(f) for f in glob.glob("/kaggle/input/historical-flight-and-weather-data/*.csv") ])
df.head()
df.dtypes
df.hist(figsize=(20,20)); # Tip: put a semicolon at the end of the line to avoid printing a bunch of text output.
df.shape
df.dtypes
(df.arrival_delay > 0).sum() / df.shape[0]
(df.arrival_delay > 30).sum() / df.shape[0]
(df.arrival_delay > 60).sum() / df.shape[0]
(df.departure_delay > 0).sum() / df.shape[0]
((df.arrival_delay > 0) & (df.departure_delay > 0)).sum() / df.shape[0]
df.cancelled_code.value_counts()
(df.cancelled_code != "N").sum() / df.shape[0]
df_cancel = df[df.cancelled_code != "N"]

df_cancel.hist(figsize=(20,20)); 
df_nocancel = df[df.cancelled_code == "N"]

df_nocancel.hist(figsize=(20,20)); 
print(df_cancel.HourlyWindSpeed_x.mean(), df_cancel.HourlyWindSpeed_x.median(), df_cancel.HourlyWindSpeed_x.max())

print(df_nocancel.HourlyWindSpeed_x.mean(), df_nocancel.HourlyWindSpeed_x.median(), df_nocancel.HourlyWindSpeed_x.max())
num_flights = df.groupby(by=["origin_airport", "destination_airport"]).size()



num_flights.head()
pd.DataFrame(num_flights.reset_index()).dtypes
num_flights = num_flights.reset_index()



num_flights.columns = ['origin_airport','destination_airport','num_flights']



g = nx.DiGraph()



for _, edge in num_flights.iterrows():

    g.add_edge(edge['origin_airport'], edge['destination_airport'], weight=edge['num_flights'])

nx.draw(g)
deg_cen = nx.degree_centrality(g)



df_deg_cen = pd.DataFrame(deg_cen.items())

df_deg_cen.columns = ["airport", "deg_cen"]



df_deg_cen.head()
bet_cen = nx.betweenness_centrality(g, weight="weight")



df_bet_cen = pd.DataFrame(bet_cen.items())

df_bet_cen.columns = ["airport", "bet_cen"]



df_bet_cen.head()
df_bet_cen.set_index("airport", inplace=True)

df_deg_cen.set_index("airport", inplace=True)





net_stats = df_bet_cen

net_stats["deg_cen"] = df_deg_cen.deg_cen



net_stats.head()
net_stats.reset_index(inplace=True)



df_net_stats = df.merge(net_stats, left_on="origin_airport", right_on="airport")



df_net_stats["origin_bet_cen"] = df_net_stats["bet_cen"]

df_net_stats["origin_deg_cen"] = df_net_stats["deg_cen"]

df_net_stats.drop(["airport", "deg_cen", "bet_cen"], inplace=True, axis=1)



df_net_stats.head()
df_net_stats = df_net_stats.merge(net_stats, left_on="destination_airport", right_on="airport")



df_net_stats["destination_bet_cen"] = df_net_stats["bet_cen"]

df_net_stats["destination_deg_cen"] = df_net_stats["deg_cen"]

df_net_stats.drop(["airport", "deg_cen", "bet_cen"], inplace=True, axis=1)



df_net_stats.head()
df_net_stats[["arrival_delay", "destination_bet_cen","destination_deg_cen", "origin_bet_cen","origin_deg_cen"]].corr()