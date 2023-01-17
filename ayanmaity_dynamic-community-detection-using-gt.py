# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import networkx as nx

import community

import matplotlib.pyplot as plt

from sklearn.metrics.cluster import normalized_mutual_info_score
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
def merge_comm(pi,G):

    comm_list = list(set(pi.values()))

    prev_pi = pi.copy()

    prev_mod = Q(pi,G)   ### here

    ##print('merge :',prev_mod)

    mod = prev_mod

    for s1 in comm_list:

        for s2 in comm_list:

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

    ##print('time: ',t,' , merge :',mod)

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

    print(q_all)

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
def CDG(G,pi,nIter=30):

    prev_pi = pi

    t = 0

    while(True):

        if t == nIter:

            break

        for u in G.nodes():

            pi =  comm_switch_ind(u,G,pi)

        pi = merge_comm(pi,G)

        t = t+1

    return pi
def form_initial_graph(graph_df,time_col,node_cols):

    time_uniq = graph_df[time_col].unique()

    graph_dict = graph_df.groupby([time_col]).groups

    init_df = graph_df.iloc[list(graph_dict[time_uniq[0]])]

    graph_init = nx.from_pandas_edgelist(init_df,node_cols[0],node_cols[1])

    return graph_init,time_uniq,graph_dict
def dyn_CDG(graph_df,time_col,node_cols,nIter=1):

    G_init,time_uniq,G_dict = form_initial_graph(graph_df,time_col,node_cols)

    pi = {v:v for v in list(G_init.nodes())}

    pi = CDG(G_init,pi,nIter)

    print(time_uniq)

    Q_list = []

    pi_list = []

    c = 0

    for t in time_uniq[1:]:

        row_list = list(G_dict[t])

        ##print(row_list)

        temp_df = graph_df.iloc[row_list]

        edge_list = list(zip(temp_df['v1'],temp_df['v2']))

        G_init.add_edges_from(edge_list)

        for v in G_init:

            if v not in pi:

                pi[v] = v

        pi = CDG(G_init,pi,nIter)

        if c%100 == 0 or t==time_uniq[-1]:

            print(len(list(G_init.nodes())),' ',len(list(pi.keys())))

            Q_list.append(Q(pi,G_init))

            pi_list.append(pi)

        c = c+1

    return G_init,pi,Q_list,pi_list
RMgraph = pd.read_csv('../input/rmgraph/out.mit',delimiter=r"\s+",header=None,skiprows=[0],index_col=False,names=['v1','v2','w','time'])

RMgraph_new = RMgraph.drop_duplicates(subset=['v1','v2'])

RMgraph_new = RMgraph_new.reset_index(drop=True)

st1 = time.time()

G,pi,Q_list,pi_list = dyn_CDG(RMgraph_new,time_col='time',node_cols=['v1','v2'],nIter=1)

end1 = time.time()
comms = []

for v in  list(pi.keys()):

    comms.append(pi[v])

comms = list(set(comms))

comms
def static_CDG_dynGraph(graph_df,time_col,node_cols,nIter=2):

    G_init,time_uniq,G_dict = form_initial_graph(graph_df,time_col,node_cols)

    pi = {v:v for v in list(G_init.nodes())}

    pi = CDG(G_init,pi,nIter)

    Q_list = []

    pi_static_list = []

    c = 0

    for t in time_uniq[1:]:

        row_list = list(G_dict[t])

        print(row_list)

        temp_df = graph_df.iloc[row_list]

        edge_list = list(zip(temp_df['v1'],temp_df['v2']))

        G_init.add_edges_from(edge_list)

        if c%100 == 0  or t==time_uniq[-1]:

            pi = {v:v for v in list(G_init.nodes())}

            pi = CDG(G_init,pi,nIter)

            Q_list.append(Q(pi,G_init))

            pi_static_list.append(pi)

        c = c+1

    return G_init,pi,Q_list,pi_static_list
st2 = time.time()

G_init,pi_stdy,Q_list_stdy,pi_static_list = static_CDG_dynGraph(RMgraph_new,time_col='time',node_cols=['v1','v2'],nIter=5)

end2 = time.time()
from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))

values = [pi.get(node) for node in G.nodes()]

cl = 0

for c in set(values):

    for i in range(len(values)):

        if values[i] == c:

            values[i] = cl

    cl = cl+1

pos = nx.spring_layout(G)

nx.draw_spring(G, node_color = values, node_size=500, with_labels=True)
plt.figure(figsize=(8,8))

values = [pi_stdy.get(node) for node in G.nodes()]

cl = 0

for c in set(values):

    for i in range(len(values)):

        if values[i] == c:

            values[i] = cl

    cl = cl+1



nx.draw_spring(G, node_color = values, node_size=500, with_labels=True)
part = community.best_partition(G,random_state=9)

community.modularity(part,G)
plt.figure(figsize=(8,8))

values = [part.get(node) for node in G.nodes()]

cl = 0

for c in set(values):

    for i in range(len(values)):

        if values[i] == c:

            values[i] = cl

    cl = cl+1



nx.draw_spring(G, node_color = values, node_size=500, with_labels=True)
def best_partition_comm(graph_df,time_col,node_cols):

    G,time_uniq,G_dict = form_initial_graph(graph_df,time_col,node_cols)

    mod_list = []

    c = 0

    part_list = []

    for t in time_uniq[1:]:

        row_list = list(G_dict[t])

        temp_df = graph_df.iloc[row_list]

        edge_list = list(zip(temp_df['v1'],temp_df['v2']))

        G.add_edges_from(edge_list)

        if c%100 == 0 or t==time_uniq[-1]:

            part = community.best_partition(G,random_state=9)

            mod = community.modularity(part,G)

            mod_list.append(mod)

            part_list.append(part)

        c = c+1

    return G,part,mod_list,part_list
G,part,mod_list,part_list = best_partition_comm(RMgraph_new,time_col='time',node_cols=['v1','v2'])
plt.figure(figsize=(12,8))

plt.plot(list(range(len(Q_list))),Q_list,'--',color='red',label='Dynamic CDGT')

plt.plot(list(range(len(Q_list_stdy))),Q_list_stdy,'--',color='blue',label='Static CDGT')

plt.plot(list(range(len(Q_list))),mod_list,'--',color='green',label='Modularity Max CD')

plt.xlabel('Time')

plt.ylabel('Modularity Value')

plt.legend()

plt.show()
from sklearn.metrics.cluster import normalized_mutual_info_score

nmi_list1 = []

nmi_list2 = []

for t in range(len(pi_list)-1):

    nmi_list1.append(normalized_mutual_info_score(list(pi_list[t+1].values()),list(pi_static_list[t+1].values())))

    nmi_list2.append(normalized_mutual_info_score(list(pi_list[t+1].values()),list(part_list[t+1].values())))
plt.figure(figsize=(12,8))

plt.plot(list(range(len(nmi_list1))),nmi_list1,'-.',color='red',label='Dynamic CDGT vs static CDGT')

plt.plot(list(range(len(nmi_list2))),nmi_list2,'--',color='blue',label='Dynamic CDGT vs Best partition')

plt.xlabel('Time')

plt.ylabel('NMI Value')

plt.legend()

plt.ylim(0,1)

plt.show()
print(end1-st1)

print(end2-st2)

print(end3-st3)
for p in pos:

    if pos[p][0]>=0.5:

        pos[p][0] = 0.5

    if pos[p][0]<= -0.5:

        pos[p][0] = -0.5

    if pos[p][1]>=0.5:

        pos[p][1] = 0.5

    if pos[p][1]<=-0.5:

        pos[p][1] = -0.5
from matplotlib.pyplot import pause

from IPython.display import clear_output

def dyn_CDG_plot(graph_df,time_col,node_cols,pos,pi_list):

    G_init,time_uniq,G_dict = form_initial_graph(graph_df,time_col,node_cols)

    c = 0

    Iter = 0

    for t in time_uniq[1:]:

        row_list = list(G_dict[t])

        ##print(row_list)

        temp_df = graph_df.iloc[row_list]

        edge_list = list(zip(temp_df['v1'],temp_df['v2']))

        G_init.add_edges_from(edge_list)

        if c%100 == 0 :

            clear_output(wait=True)

            plt.figure(figsize=(12,10))

            plt.xlim(-0.6,0.6)

            plt.ylim(-0.6,0.6)

            plt.title('Time : '+str(c))

            values = [pi_list[Iter].get(node) for node in G_init.nodes()]

            nx.draw_networkx(G_init, pos = pos,node_color = values, node_size=500, with_labels=True)

            plt.savefig('network_'+str(c)+'.png',format='PNG')

            pause(2)

            Iter = Iter+1

        if  t==time_uniq[-1] :

            clear_output(wait=True)

            plt.figure(figsize=(12,10))

            plt.xlim(-0.6,0.6)

            plt.ylim(-0.6,0.6)

            plt.title('Time : '+str(c))

            values = [pi_list[-1].get(node) for node in G_init.nodes()]

            nx.draw_networkx(G_init,pos = pos, node_color = values, node_size=500, with_labels=True)

            plt.savefig('network_'+str(c)+'.png',format='PNG')

            pause(2)

        c = c+1
dyn_CDG_plot(RMgraph_new,time_col='time',node_cols=['v1','v2'],pos=pos,pi_list=pi_list)