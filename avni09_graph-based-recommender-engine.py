import pandas as pd

import numpy as np

import networkx as nx
# find neibhours for a given node
fb_network = pd.read_csv("../input/fb_network_sub.csv")
G = nx.from_pandas_edgelist(fb_network, source="source_node", target="destination_node")

# how many friends does user 1 has

len(list(G.neighbors(741)))
# same as 

val1 = fb_network.loc[fb_network["source_node"] == 741,"destination_node"].values

val2 = fb_network.loc[fb_network["destination_node"] == 741, "source_node"].values
len(set(list(np.append(val1,val2))))
## Node centrality

fb_deg_cen = nx.degree_centrality(G)
type(fb_deg_cen)
fb_deg_cen[741]
# same as 

len(list(G.neighbors(741)))/(len(list(G.nodes())) - 1)
fb_deg_cen = pd.DataFrame([fb_deg_cen]).T
fb_deg_cen["user_id"] = fb_deg_cen.index

fb_deg_cen = fb_deg_cen.rename(columns={0:"deg_cen"})
fb_deg_cen.head()
fb_deg_cen[fb_deg_cen["user_id"] == 741]
fb_deg_cen.sort_values(by="deg_cen", ascending=False).head()
len(list(G.neighbors(1492489)))/(len(list(G.nodes())) - 1)
## find few unconnected people in the network

count = 0

for e in nx.non_edges(G):

    print(e)

    count = count + 1

    if count > 10:

        break
nx.shortest_path(G, source = 786432, target = 1048579)
## common neighbours

## find common neighbours for a paticular user say user no 4850

len(list(G.neighbors(4850)))
# find list of key users the user 4850 is not currently connected to

key_users = list(set(fb_network["source_node"].values))

unconnected_users = [x for x in key_users if x not in list(G.neighbors(4850))]
len(unconnected_users)
# find common neighbours

common_neighbors = []

for user_id in unconnected_users:

    common_neighbors.append(len(list(nx.common_neighbors(G, 4850, user_id))))
common_neighbors_df = pd.DataFrame({"user_id": 4850, 

                                    "unconnctd_user_id": unconnected_users, "common_neighbors": common_neighbors})
common_neighbors_df.sort_values(by = "common_neighbors", ascending=False).head()
n1 = set(G.neighbors(4850))

n2 = set(G.neighbors(1492489))
len(n1.intersection(n2))
list(nx.jaccard_coefficient(G, [(4850,1492489)]))
len(n1.intersection(n2))/len(n1.union(n2))
jac_sim = [list(nx.jaccard_coefficient(G, [(4850,user_id)]))[0] for user_id in unconnected_users]
jac_sim_df =  pd.DataFrame.from_records(jac_sim,columns=["user_id","unconnctd_user_id","jac_sim"])
jac_sim_df.sort_values(by="jac_sim", ascending=False).head()
n1 = set(G.neighbors(4850))

n2 = set(G.neighbors(245648))
print("intersection -" ,len(n1.intersection(n2)))

print("union -" , len(n1.union(n2)))
40/1026
list(nx.resource_allocation_index(G,[(4850,1492489)]))
res_alloc_sim = [list(nx.resource_allocation_index(G, [(4850,user_id)]))[0] for user_id in unconnected_users]
res_alloc_sim_df =  pd.DataFrame.from_records(res_alloc_sim,columns=["user_id","unconnctd_user_id","res_alloc_sim"])
res_alloc_sim_df.sort_values(by="res_alloc_sim", ascending=False).head()