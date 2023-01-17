%%capture

!pip install networkx==2.3
import pandas as pd

import networkx as nx

#%matplotlib notebook

import matplotlib.pyplot as plt

import os

import operator

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))
G_df = pd.read_csv('../input/openflights-route-database-2014/routes.csv')

#Count_df = pd.read_csv('D:/Personal and studies/College/Semester 6/Social and information networks project/countries.txt')

cols_list=["City","Country","IATA"]

airport_df = pd.read_csv('../input/openflights-airports-database-2017/airports.csv',usecols=cols_list)
G_df.head(2)
#Count_df.head(2)
airport_df.head(2)
G_draw = nx.from_pandas_edgelist(G_df.head(1000), 'Source airport', 'Destination airport',create_using=nx.DiGraph())
plt.figure(figsize=(12,8))

nx.draw(G_draw,pos=nx.spring_layout(G_draw),with_labels=False)
G = nx.from_pandas_edgelist(G_df, 'Source airport', 'Destination airport',create_using=nx.DiGraph())
print(nx.info(G))
#does a route exist between every two airport? #is every airport reachable from every other airport?

nx.is_strongly_connected(G), nx.is_connected(G.to_undirected())
#How many nodes are in the largest (in terms of nodes) weakly connected component?

wccs = nx.weakly_connected_components(G)

x=len(max(wccs, key=len))

print(x)

print(x/len(G.nodes()))



#so means 99% of graph is weakly connected
#How many nodes are in the largest (in terms of nodes) strongly connected component?

sccs = nx.strongly_connected_components(G)

x=len(max(sccs, key=len))

print(x)

print(x/len(G.nodes()))

#so 97% are strongly connected 
scc_subs = nx.strongly_connected_component_subgraphs(G)

G_sc = max(scc_subs, key=len) #the largest strongly connected subgraph

shortest_sc=nx.average_shortest_path_length(G_sc)

shortest_sc
wcc_subs = nx.weakly_connected_component_subgraphs(G)

G_wc = max(wcc_subs, key=len) #the largest weakly connected subgraph

shortest_wc=nx.average_shortest_path_length(G_wc)

shortest_wc
len(G_sc.edges())/len(G_sc.nodes()) 
len(G_wc.edges())/len(G_wc.nodes())
nx.density(G),nx.density(G_sc)
degrees = dict(G.degree())

degree_values = sorted(set(degrees.values()))

histogram = [list(degrees.values()).count(i)/float(nx.number_of_nodes(G_sc)) for i in degree_values]
plt.plot(histogram)
diameter=nx.diameter(G_sc)

diameter
radius=nx.radius(G_sc)

radius
per=nx.periphery(G_sc)

per
airport_df.loc[airport_df['IATA'].isin(per)]
cen=nx.center(G_sc)

cen
airport_df.loc[airport_df['IATA'].isin(cen)]
max_count = -1

result_node = None

for node in per:

    count = 0

    sp = nx.shortest_path_length(G_sc, node)

    for key, value in sp.items():

        if value == diameter:

            count += 1        

    if count > max_count:

        result_node = node

        max_count = count



result_node, max_count
airport_df.loc[airport_df['IATA'] == result_node]
d = radius

max_count = -1

result_node = None

for node in cen:

    count = 0

    sp = nx.shortest_path_length(G_sc, node)

    for key, value in sp.items():

        if value == radius:

            count += 1        

    if count > max_count:

        result_node = node

        max_count = count



result_node, max_count
airport_df.loc[airport_df['IATA'] == result_node]
node = result_node

list(nx.minimum_node_cut(G_sc, cen[0], node)),len(nx.minimum_node_cut(G_sc, cen[0], node))
nx.transitivity(G_sc), nx.average_clustering(G_sc)
nx.transitivity(G_wc), nx.average_clustering(G_wc)
in_deg=nx.in_degree_centrality(G_sc)
top5=sorted(in_deg.items(), key=operator.itemgetter(1),reverse=True)[:5]

l=[]

for i,j in top5:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
bot5=sorted(in_deg.items(), key=operator.itemgetter(1))[:5]

l=[]

for i,j in bot5:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
out_deg=nx.out_degree_centrality(G_sc)
top5=sorted(out_deg.items(), key=operator.itemgetter(1),reverse=True)[:5]

top5
l=[]

for i,j in top5:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
bot5=sorted(out_deg.items(), key=operator.itemgetter(1))[:5]

bot5
l=[]

for i,j in bot5:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
closeness = nx.closeness_centrality(G_sc, wf_improved=True)
close=sorted(closeness.items(), key=operator.itemgetter(1),reverse=True)[:5]

l=[]

for i,j in close:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
close=sorted(closeness.items(), key=operator.itemgetter(1))[:18]

l=[]

for i,j in close:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
betweeness = nx.betweenness_centrality(G_sc, normalized=True)
close=sorted(betweeness.items(), key=operator.itemgetter(1),reverse=True)[:5]

l=[]

for i,j in close:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
close=sorted(betweeness.items(), key=operator.itemgetter(1))[:5]

l=[]

for i,j in close:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
arti=list(nx.articulation_points(G_sc.to_undirected()))

len(arti)
nx.has_bridges(G.to_undirected())
len(list(nx.bridges(G.to_undirected())))
pr = nx.pagerank(G_sc, alpha=0.85)
pager=sorted(pr.items(), key=operator.itemgetter(1),reverse=True)[:5]

l=[]

for i,j in pager:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
pager=sorted(pr.items(), key=operator.itemgetter(1))[:5]

l=[]

for i,j in pager:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
hits = nx.hits(G_sc)
hubs=sorted(hits[0].items(), key=operator.itemgetter(1))[:5]

l=[]

for i,j in hubs:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
auth=sorted(hits[1].items(), key=operator.itemgetter(1))[:5]

l=[]

for i,j in auth:

    l.append(i)

airport_df.loc[airport_df['IATA'].isin(l)]
degrees = dict(G_sc.degree())

degree_values = sorted(set(degrees.values()))

histogram = [list(degrees.values()).count(i)/float(nx.number_of_nodes(G_sc)) for i in degree_values]
plt.plot(histogram)
df = pd.DataFrame(index=G_sc.nodes())

df['clustering'] = pd.Series(nx.clustering(G_sc))

df['in_degree'] = pd.Series(dict(in_deg))



df['out_degree'] = pd.Series(dict(out_deg))

df['degree_centrality'] = pd.Series(nx.degree_centrality(G))

df['closeness'] = pd.Series(closeness)
df['betweeness'] = pd.Series(betweeness)

df['pr'] = pd.Series(pr)

df['hits_hubs'] = pd.Series(hits[0])

df['hits_auth'] = pd.Series(hits[1])
df.head()