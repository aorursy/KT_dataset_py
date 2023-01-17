# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import networkx as nx
import community
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
G = nx.karate_club_graph()
nx.draw_spring(G, with_labels=True)
def q(u,pi,G):
    s = pi[u]
    s_nodes = [k for k,v in pi.items() if v == s]
    G1 = G.subgraph(s_nodes)
    i_u = G1.degree(u)
    d_u = G.degree(u)
    D_s = sum(list((dict(G.degree(s_nodes)).values())))
    m = sum(list((dict(G.degree()).values())))
    return i_u - ((d_u*D_s)/m)
def q_list(u,comm,G):
    G1 = G.subgraph(comm)
    i_u = G1.degree(u)
    d_u = G.degree(u)
    D_s = sum(list((dict(G.degree(comm)).values())))
    m = sum(list((dict(G.degree()).values())))
    return i_u - ((d_u*D_s)/m)
def Q(pi,G):
    Q_total = 0
    for u in G.nodes():
        Q_total = Q_total + q(u,pi,G)
    m = sum(list((dict(G.degree()).values())))
    return Q_total/m
def comm_switch_ind(u,G,pi):
    V = list(G[u])
    V_comm = list(set([pi[v] for v in V]))
    prev_pi = pi.copy()
    prev_mod = q(u,pi,G)  ### here
    ###print(prev_mod)
    mod = prev_mod
    for comm in V_comm:
        pi[u] = comm
        new_mod = q(u,pi,G) ### here
        ###print(comm,' ',new_mod)
        if new_mod > mod:
            mod = new_mod
            mod_comm = comm
    ###print(mod)
    if mod > prev_mod:
        prev_pi[u] = mod_comm
        return prev_pi
    else:
        return prev_pi
def n_comms(pi,G,s1):
    s1_nodes = [k for k,v in pi.items() if v == s1]
    s2_nodes= []
    for u in s1_nodes:
        s2_nodes = s2_nodes+list(G[u])
    s2_nodes = list(set(s2_nodes))
    s2list = []
    for v in s2_nodes:
        s2list.append(pi[v])
    s2list = list(set(s2list))
    return s2list
def merge_comm(pi,G):
    comm_list = list(set(pi.values()))
    prev_pi = pi.copy()
    prev_mod = Q(pi,G)   ### here
    ##print('merge :',prev_mod)
    mod = prev_mod
    for s1 in comm_list:
        ncomm = n_comms(prev_pi,G,s1)
        for s2 in ncomm:
            if s1!=s2 :
                pi = prev_pi.copy()
                s1_nodes = [k for k,v in pi.items() if v == s1]
                for u in s1_nodes:
                    pi[u] = s2
                new_mod = Q(pi,G)    ### here
                if new_mod > mod:
                    ##print(s1,s2)
                    ##print(list(set(pi.values())))
                    mod = new_mod
                    prev_pi = pi.copy()
    ##print('merge :',mod)
    return prev_pi
def list_pi(pi):
    pi_list = {}
    for v in list(pi.keys()):
        pi_list[v] = [pi[v]]
    return pi_list
def comm_list(pi_list,comm):
    node_list = []
    for v in pi_list.keys():
        if comm in pi_list[v]:
            node_list.append(v)
    return node_list
import statistics
def q_std(G,pi):
    q_all = []
    for v in list(G.nodes()):
        q_all.append(q(v,pi,G))
    ##print(q_all)
    return q_all,statistics.stdev(q_all),statistics.mean(q_all)
def overlap_switch(u,pi_list,G,q_all,th):
    comms = []
    for v in  list(pi_list.keys()):
        comms.extend(pi_list[v])
    comms = list(set(comms)-set(pi_list[u]))
    
    for comm in comms:
        node_list = comm_list(pi_list,comm)
        node_list.append(u)
        new_mod = q_list(u,node_list,G) ### here
        ###print(comm,' ',new_mod)
        if new_mod > q_all[u]-0.5*th and new_mod > 0:
            pi_list[u].append(comm)
    return pi_list
def CDG(G,nIter=30):
    pi = {v:v for v in list(G.nodes())}
    prev_pi = pi
    t = 0
    while(True):
        if t == nIter:
            break
        for u in G.nodes():
            pi =  comm_switch_ind(u,G,pi)
        pi = merge_comm(pi,G)
        if normalized_mutual_info_score(list(prev_pi.values()),list(pi.values()))==1:
            print('yes')
            break
        prev_pi = pi
        t = t+1
    print("Iter "+str(t))
    """q_all,q_stdev,q_mean = q_std(G,pi)
    thres = q_stdev
    ##print("q_mean,q_stedev,thres : ",q_mean,q_stdev,thres)
    pi_list = list_pi(pi)
    for u in G.nodes():
        pi =  overlap_switch(u,pi_list,G,q_all,thres)"""
    return pi
G_new = nx.random_geometric_graph(400, .20)
nx.draw_spring(G_new,with_labels=True)
pi_new = CDG(G_new,50)
values = [pi_new.get(node) for node in list(G_new.nodes())]
cl = 0
for c in set(values):
    for i in range(len(values)):
        if values[i] == c:
            values[i] = cl
    cl = cl+1
cl
pos = nx.get_node_attributes(G_new, 'pos')

# find node near center (0.5,0.5)
dmin = 1
ncenter = 0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5)**2 + (y - 0.5)**2
    if d < dmin:
        ncenter = n
        dmin = d

# color by path length from node near center
p = dict(nx.single_source_shortest_path_length(G_new, ncenter))

plt.figure(figsize=(10, 10))
nx.draw_networkx_edges(G_new, pos,  alpha=0.4)
nx.draw_networkx_nodes(G_new, pos,
                       node_size=500,
                       node_color=values,
                       cmap=plt.cm.RdYlGn
                       )
nx.draw_networkx_labels(G_new,pos)
                       
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis('off')
plt.show()
comms = []
for v in  list(pi_new.keys()):
    comms.append(pi_new[v])
comms = list(set(comms))
comms
Q(pi_new,G_new)
part = community.best_partition(G_new,random_state=9)
community.modularity(part,G_new)
values = [part.get(node)*1000 for node in G_new.nodes()]
pos = nx.get_node_attributes(G_new, 'pos')

# find node near center (0.5,0.5)
dmin = 1
ncenter = 0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5)**2 + (y - 0.5)**2
    if d < dmin:
        ncenter = n
        dmin = d

# color by path length from node near center
p = dict(nx.single_source_shortest_path_length(G_new, ncenter))

plt.figure(figsize=(10, 10))
nx.draw_networkx_edges(G_new, pos,  alpha=0.4)
nx.draw_networkx_nodes(G_new, pos,
                       node_size=500,
                       node_color=values,
                       cmap=plt.cm.RdYlGn
                       )
nx.draw_networkx_labels(G_new,pos)
                       
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis('off')
plt.show()
normalized_mutual_info_score(list(pi_new.values()),list(part.values()))
from networkx.algorithms.community.centrality import girvan_newman
ng_comm = girvan_newman(G_new)

