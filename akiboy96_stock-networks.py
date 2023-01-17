# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import csv
import os
import re

# %load_ext rpy2.ipython
# import rpy2.rinterface
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_stockes = pd.read_csv("../input/usstockprices/stocks_price_final.csv", index_col = 0)
df_SP500 = pd.read_csv("../input/sp500-symbols/SP500.csv")
symbols = df_SP500['symbol'].tolist()
companies=np.random.choice(symbols, size=500, replace=False)
#Convert matket.cap from string to float
mapping = dict(K='', M='', B='')
df_stockes["market.cap"] = pd.to_numeric(df_stockes['market.cap'].str.strip('$').replace(mapping, regex=True))
print(df_stockes.head())
print(df_stockes.shape)
#Filter by SP500
df_stockes=df_stockes[df_stockes['symbol'].isin(companies)]
df_stockes.head()
df_stockes.shape
#df_stock_list = df_stockes[df_stockes[]]
df_stock_list = df_stockes[(df_stockes.symbol != 'GOOG') & (df_stockes.symbol != 'FOX') & (df_stockes.symbol != 'UA') & (df_stockes.symbol != 'DISCK') & (df_stockes.symbol != 'NWS')]
print(df_stock_list.shape)
#df_stockes = df_stockes[(df_stockes.symbol != 'GOOG')]
#df_stockes.shape
#df_stock_list = df_stockes.head()
#Filter by Market Cap|
# df_stockes = df_stockes[df_stockes['market.cap'] > 2.0]
col_close_price = 'adjusted'
df_stock_prices = df_stock_list[['symbol','date', col_close_price]]
# Remove duplicate if any and keep the latest data
df_stock_prices = df_stock_prices.drop_duplicates( keep='last')
# Format the date field
df_stock_prices['date'] = pd.to_datetime(df_stock_prices['date'], format='%Y%m%d', errors='ignore')
df_stock_prices.set_index(['date','symbol'],inplace=True)
# Change to wide format. Index is date and Columns are symbols. Values are adjsuted price
df_stock_prices=df_stock_prices.unstack()[col_close_price]
df_stock_prices.reset_index(inplace=True)
#Replace Null with adjacent values 
df_stock_prices.fillna(method='bfill',inplace=True)
df_stock_prices.fillna(method='ffill',inplace=True)
# Calculate daily log returns
T = 1 #Daily return
#Skip first columns which is date
for column in df_stock_prices.columns.values.tolist()[1:]:
#     print(column)
    df_stock_prices[column] = np.log(df_stock_prices[column]) - np.log(df_stock_prices[column].shift(T))

df_stock_prices.set_index('date',inplace=True)
df_stock_prices.fillna(method='bfill',inplace=True)
df_stock_prices.fillna(method='ffill',inplace=True)
#Plot daily log returns
%matplotlib inline
fig, ax1 = plt.subplots(figsize=(20, 15))
df_stock_prices.plot(ax=ax1, legend=False)
# plt.ylim([-0.5, 0.5])
plt.tight_layout()
plt.show()
len(df_stock_prices)
# Function to calculate corr
def calculate_corr(df_stock_returns, returns_window, corr_window_size, corr_method):
    stocks_cross_corr_dict = {}
    #Calculate mean correlation by window for plot
    x_days = []
    y_mean_corr = []        
#     W = corr_window_size
    for i in range(returns_window,len(df_stock_returns),corr_window_size):
        dic_key = i
        stocks_cross_corr_dict[dic_key]=df_stock_returns.iloc[i:(i+W)].corr(method='pearson')
        stocks_cross_corr_dict[dic_key].fillna(0,inplace=True)
        x_days.append(dic_key)
        y_mean_corr.append(np.mean([abs(j) for j in stocks_cross_corr_dict[dic_key].values.flatten().tolist()]))        
    return stocks_cross_corr_dict, x_days,y_mean_corr
# Plot corr for various windows
%matplotlib inline
# stocks_cross_corr_dict = {}
#Time Window width
#TO DO: try different windows and differnt algorithms
#t= 21 #21 based on the paper Asset trees and asset graphs in financial markets J.-P. Onnela et all
# Try window from 1 month to 6 months of trading days
# 21 days is one month trading days
start = 21
end = 126
step = 21;
plt.figure(figsize=(20, 10))
#Find corr for the entire time period 
# _, x_days, y_mean_corr = calculate_corr(df_stock_prices,1,len(df_stock_prices), 'pearson')
# x_days_t = range(0,len(df_stock_prices), 1)
# y_mean_corr_t = np.empty(len(df_stock_prices))
# y_mean_corr_t.fill(y_mean_corr[0])
# plt.plot(x_days_t, y_mean_corr_t)
for t in range(start, end, step):
    x_days = []
    y_mean_corr = []
    W = t
    _, x_days, y_mean_corr = calculate_corr(df_stock_prices,1,W, 'pearson')
    plt.plot(x_days, y_mean_corr)
    plt.xlabel('Days')
    plt.ylabel('Mean Correlation')
    l = list(range(start, end, step))
#     l.insert(0, len(df_stock_prices))
    plt.legend(l, loc='upper left')     

plt.show()
#Calculate corr for the entire period.
stocks_cross_corr, _, _ = calculate_corr(df_stock_prices,1, len(df_stock_prices), 'pearson')
stocks_cross_corr[1]
#Build the Graph with stocks as nodes and corr as edges
import networkx as nx
import networkx.algorithms.community as nxcom
import community

edge_weights = []
def build_graph(stocks_cross_corr, threshold):
    graph_edges = []
    for x in stocks_cross_corr.keys():
        for y in stocks_cross_corr[x].keys():
            #print(x, y) 
            # Filter by absolute value of the corr
            if abs(stocks_cross_corr[x][y]) > threshold:
                #if same stock, continue
                if  x == y:
                    continue
                if x < y: #Avoid duplicates, AxAAL vs AALxA
                    graph_edges.append([x,y,dict(weight=abs(stocks_cross_corr[x][y]))])
                    edge_weights.append(abs(stocks_cross_corr[x][y]))
                else:
                    None
    
#   print(len(graph_edges))
    G = nx.Graph()
    G.add_edges_from(graph_edges)
    return G
#     partition = community.best_partition(G)
#     modularity = community.modularity(partition, G)
#     values = [partition.get(node) for node in G.nodes()]
#     nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
#     print(modularity)    
    
import networkx as nx
import networkx.algorithms.community as nxcom
import community


stocks_cross_corr, _, _ = calculate_corr(df_stock_prices,1, len(df_stock_prices), 'pearson')
stocks_cross_corr = stocks_cross_corr[1]

corr_thresholds = np.linspace(0.5, 0.95, 20)
modularity_list = []
community_list = []
for cor in corr_thresholds:
    G = build_graph(stocks_cross_corr, cor)
    partition = community.best_partition(G)
    modularity = community.modularity(partition, G)
    modularity_list.append(modularity)
    community_list.append(len(G.nodes()))
    

#

# partition = community.best_partition(G)
# modularity = community.modularity(partition, G)
# values = [partition.get(node) for node in G.nodes()]
# nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
# print(modularity)
# print("Total number of Communities=", len(G.nodes()))

# partition=community.best_partition(G)
# # Calculating modularity and the total number of communities
# mod=community.modularity(partition,G)
# print("Modularity: ", mod)
# print("Total number of Communities=", len(G_comm.nodes()))

# dict_degree_centrality = nx.degree_centrality(G)
# dict_closeness_centrality = nx.closeness_centrality(G)
# dict_eigenvector_centrality = nx.eigenvector_centrality(G)
# print("dict_degree_centrality: ", dict_degree_centrality)
# print("dict_closeness_centrality: ", dict_closeness_centrality)
# print("dict_eigenvector_centrality: ", dict_eigenvector_centrality)
# Louvian
%matplotlib inline
stocks_cross_corr, _, _ = calculate_corr(df_stock_prices,1, len(df_stock_prices), 'pearson')
stocks_cross_corr = stocks_cross_corr[1]

cor_thresold = 0.6
G = build_graph(stocks_cross_corr, cor_thresold)
partition = community.best_partition(G)
modularity = community.modularity(partition, G)
values = [partition.get(node) for node in G.nodes()]
plt.figure(figsize=(10,10))
nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
print(modularity)
print("Total number of Communities=", len(G.nodes()))

dict_betwenness_centrality = nx.betweenness_centrality(G)
dict_degree_centrality = nx.degree_centrality(G)
dict_closeness_centrality = nx.closeness_centrality(G)
dict_eigenvector_centrality = nx.eigenvector_centrality(G)
print("dict_degree_centrality: ", dict_degree_centrality)
print("dict_closeness_centrality: ", dict_closeness_centrality)
print("dict_eigenvector_centrality: ", dict_eigenvector_centrality)
print("dict_betweenness_centrality: ", dict_betwenness_centrality)

#Portfolio Formula: 
c_dict = dict([(k, [dict_betwenness_centrality[k], dict_eigenvector_centrality[k], dict_degree_centrality[k], dict_closeness_centrality[k] ]) for k in dict_betwenness_centrality])
#print(c_dict)    
    
C_total = {}
for key in c_dict: 
    C_total[key] = sum(c_dict[key]) 
        

print("The Centrality total for stocks are:", C_total)   

newDict = dict(filter(lambda elem: elem[1] > 0.3, C_total.items()))
print("Stocks greater than 0.3 centrality are",newDict)
print(len(newDict))
def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1 
            
def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0
            
def get_color(i, r_off=1, g_off=1, b_off=1):
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)
#Community detection using Girvan Newman (GN)
stocks_cross_corr, _, _ = calculate_corr(df_stock_prices,1, len(df_stock_prices), 'pearson')
stocks_cross_corr = stocks_cross_corr[1]


cor_thresold = 0.6
G = build_graph(stocks_cross_corr, cor_thresold)
result = nxcom.girvan_newman(G)
communities_gn = next(result)
# Set node and edge communities
set_node_community(G, communities_gn)
set_edge_community(G)
print("GN Communities: ", len(communities_gn))

# Set community color for nodes
node_color = [    
    get_color(G.nodes[v]['community'])    
    for v in G.nodes]

# Set community color for internal edgese
external = [    
    (v, w) for v, w in G.edges    
    if G.edges[v, w]['community'] == 0]
internal = [    
    (v, w) for v, w in G.edges    
    if G.edges[v, w]['community'] > 0]
internal_color = [    
    get_color(G.edges[e]['community'])    
    for e in internal]

stock_pos = nx.spring_layout(G)
plt.rcParams.update({'figure.figsize': (15, 15)})
# Draw external edges
nx.draw_networkx(    
    G, pos=stock_pos, node_size=0,    
    edgelist=external, edge_color="#333333", with_labels=False)
# Draw nodes and internal edges
nx.draw_networkx(    
    G, pos=stock_pos, node_color=node_color,    
    edgelist=internal, edge_color=internal_color, with_labels=False)
# tuple(sorted(c) for c in next(communities_gn))
#print("List of GN Community = ", list(communities_gn))
# for communities in itertools.islice(comp, k):
#     print(tuple(sorted(c) for c in communities)) 
#Community detection using CNM
cor_thresold = 0.6
G = build_graph(stocks_cross_corr, cor_thresold)

communities_cnm = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
# Set node and edge communities
set_node_community(G, communities_cnm)
set_edge_community(G)
print("CNM Communities: ", len(communities_cnm))

# Set community color for nodes
node_color = [    
    get_color(G.nodes[v]['community'])    
    for v in G.nodes]

# Set community color for internal edgese
external = [    
    (v, w) for v, w in G.edges    
    if G.edges[v, w]['community'] == 0]
internal = [    
    (v, w) for v, w in G.edges    
    if G.edges[v, w]['community'] > 0]
internal_color = [    
    get_color(G.edges[e]['community'])    
    for e in internal]

stock_pos = nx.spring_layout(G)
plt.rcParams.update({'figure.figsize': (15, 15)})
# Draw external edges
nx.draw_networkx(    
    G, pos=stock_pos, node_size=0,    
    edgelist=external, edge_color="#333333", with_labels=False)
# Draw nodes and internal edges
nx.draw_networkx(    
    G, pos=stock_pos, node_color=node_color,    
    edgelist=internal, edge_color=internal_color, with_labels=False)
# #Community detection using Fluid Communities
# cor_thresold = 0.6
# G = build_graph(stocks_cross_corr, cor_thresold)
# number_of_communities = 10
# result = nxcom.asyn_fluidc(G, number_of_communities)
# communities_fluid = next(result)
# # Set node and edge communities
# set_node_community(G, communities_fluid)
# set_edge_community(G)
# print("Fluid Communities: ", len(communities_fluid))

# # Set community color for nodes
# node_color = [    
#     get_color(G.nodes[v]['community'])    
#     for v in G.nodes]

# # Set community color for internal edgese
# external = [    
#     (v, w) for v, w in G.edges    
#     if G.edges[v, w]['community'] == 0]
# internal = [    
#     (v, w) for v, w in G.edges    
#     if G.edges[v, w]['community'] > 0]
# internal_color = [    
#     get_color(G.edges[e]['community'])    
#     for e in internal]

# stock_pos = nx.spring_layout(G)
# plt.rcParams.update({'figure.figsize': (15, 15)})
# # Draw external edges
# nx.draw_networkx(    
#     G, pos=stock_pos, node_size=0,    
#     edgelist=external, edge_color="#333333", with_labels=False)
# # Draw nodes and internal edges
# nx.draw_networkx(    
#     G, pos=stock_pos, node_color=node_color,    
#     edgelist=internal, edge_color=internal_color, with_labels=False)
cliques = list(nx.find_cliques(G))
max_clique = max(cliques, key=len)
# Visualize maximum clique
node_color = [(0.5, 0.5, 0.5) for v in G.nodes()]
for i, v in enumerate(G.nodes()):
    if v in max_clique:
        node_color[i] = (0.5, 0.5, 0.9)
nx.draw_networkx(G, node_color=node_color, pos=stock_pos, with_labels=False)
#Create graph and write it as GraphML
stocks_cross_corr, _, _ = calculate_corr(df_stock_prices,1, len(df_stock_prices), 'pearson')
stocks_cross_corr = stocks_cross_corr[1]
cor_thresold = 0.6
G = build_graph(stocks_cross_corr, cor_thresold)

#sp_500_graph_06.graphml
#sp_500_graph_08.graphml
#stocks_2B_graph_06.graphml
# stocks_2B_graph_08.graphml
nx.write_graphml(G,'sp_500_graph_06.graphml')
stocks_cross_corr, _, _ = calculate_corr(df_stock_prices,1, len(df_stock_prices), 'pearson')
stocks_cross_corr = stocks_cross_corr[1]
cor_thresold = 0.8
G = build_graph(stocks_cross_corr, cor_thresold)

nx.write_graphml(G,'sp_500_graph_08.graphml')
# g = Graph(directed=False)
# g.add_vertices(len(edges))
# i = 0
# for x in edges:
#     g.vs[i]["id"] = x
#     g.vs[i]["label"] = x
#     i = i + 1

# # g.es["weight"] = weights
# # g.es["label"] = weights

import igraph as ig
from tabulate import tabulate

Gix08 = ig.read('sp_500_graph_08.graphml',format="graphml")
Gix06 = ig.read('sp_500_graph_06.graphml',format="graphml")

# Community detection with GN 
dendrogram = Gix06.community_edge_betweenness(directed=False)
optimal_count = dendrogram.optimal_count
print("Optimum community count: ", optimal_count)
# convert it into a flat clustering
clusters = dendrogram.as_clustering()
# get the membership vector
membership = clusters.membership
modularity = clusters.q
print("Modularity: ", modularity)
# Gix06.es[0]
# Gix.vs[0]
community_list_gn = []
for name, membership in zip(Gix06.vs["id"], membership):
    community_list_gn.append([name, membership])
#     print(name, membership)
df_community_gn = pd.DataFrame(community_list_gn, columns = ['symbol', 'community'])
# df_community.set_index('symbol',inplace=True)
# df_community.sort_values(by=['community', 'symbol'], inplace=True)
# print(df_community)
df_community_gn = df_community_gn.groupby('community', as_index=True).agg(lambda x: ', '.join(set(x.astype(str))))
print(df_community_gn.to_markdown())
import random
random.seed(1)

ig.plot(clusters, label=True, mark_groups = True)
# Community detection with CNM 
dendrogram_cnm = Gix06.community_fastgreedy(weights="weight")
optimal_count_cnm = dendrogram_cnm.optimal_count
print("CNM Optimum community count: ", optimal_count_cnm)
# convert it into a flat clustering
clusters_cnm = dendrogram_cnm.as_clustering()
# get the membership vector
membership_cnm = clusters_cnm.membership
modularity_cnm = clusters_cnm.q
print("Modularity: ", modularity)
import random
random.seed(1)

ig.plot(clusters_cnm, label=True, mark_groups = True)
community_list_cnm = []
for name, membership in zip(Gix06.vs["id"], membership_cnm):
    community_list_cnm.append([name, membership])
#     print(name, membership)
df_community_cnm = pd.DataFrame(community_list_cnm, columns = ['symbol', 'community'])
# df_community_cnm.set_index('symbol',inplace=True)
# df_community_cnm.sort_values(by=['community', 'symbol'], inplace=True)

df_community_cnm = df_community_cnm.groupby('community', as_index=True).agg(lambda x: ', '.join(set(x.astype(str))))
print(df_community_cnm.to_markdown())