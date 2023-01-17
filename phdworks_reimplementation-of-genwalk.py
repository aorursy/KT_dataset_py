import numpy as np

import networkx as nx

import time

import random

from collections import defaultdict
node_to_degree = dict()

edge_to_weight = dict()



with open("../input/deepcas-paper-dataset/global_graph.txt", 'r') as f:

    for line in f:

        line = line.rstrip('\r\n')

        parts = line.split("\t\t")

        source = int(parts[0])

        if parts[1] != "null":

            node_freq_strs = parts[1].split("\t")

            for node_freq_str in node_freq_strs:

                node_freq = node_freq_str.split(":")

                weight = int(node_freq[1])

                target = int(node_freq[0])

                edge_to_weight[(source, target)] = weight

            degree = len(node_freq_strs)

        else:

            degree = 0



        node_to_degree[source] = degree

def parse_line(line):

    line = line.rstrip('\r\n')

    parts = line.split("\t")

    edge_strs = parts[4].split(" ")

    node_to_edges = dict()

    for edge_str in edge_strs:

        edge_parts = edge_str.split(":")

        source = int(edge_parts[0])

        target = int(edge_parts[1])



        if not source in node_to_edges:

            neighbors = list()

            node_to_edges[source] = neighbors

        else:

            neighbors = node_to_edges[source]

            

        neighbors.append((target, node_to_degree.get(target, 0)))

    

    nx_G = nx.DiGraph()

    for source, nbr_weights in node_to_edges.items():

        for nbr_weight in nbr_weights:

            target = nbr_weight[0]

            edge_weight = edge_to_weight.get((source, target), 0) + 0.01 # pseudo_count



            nx_G.add_edge(source, target, weight=edge_weight)



    return [parts[0]], nx_G
def extract_first_time(nx_G):

    roots = [node for node,_ in nx_G.out_degree(weight="weight")] # List of the starting nodes.

    weights = [w if w > 0 else 0.01 for n,w in nx_G.out_degree(weight="weight")]

    weight_sum = sum(weights)

    probs = [w / weight_sum for w in  weights]



    return roots, probs



def extract(nx_G):

    roots_noleaf = [node for node,w in nx_G.out_degree(weight="weight") if w>0 ]  # List of the starting nodes excluding nodes without outgoing neighbors.    

    weights_noleaf = [w for n,w in nx_G.out_degree(weight="weight") if w > 0]

    weight_sum_noleaf = sum(weights_noleaf)

    probs_noleaf = [w / weight_sum_noleaf for w in weights_noleaf]

    

    return roots_noleaf, probs_noleaf
def alias_setup(probs):

    K = len(probs)

    q = np.zeros(K)

    J = np.zeros(K, dtype=np.int)



    smaller = []

    larger = []

    for kk, prob in enumerate(probs):

        q[kk] = K * prob

        if q[kk] < 1.0:

            smaller.append(kk)

        else:

            larger.append(kk)



    while len(smaller) > 0 and len(larger) > 0:

        small = smaller.pop()

        large = larger.pop()



        J[small] = large

        q[large] = q[large] + q[small] - 1.0

        if q[large] < 1.0:

            smaller.append(large)

        else:

            larger.append(large)



    return J, q
def get_alias_edge(nx_G, src, dst, p=0.1, q=0.1):

    unnormalized_probs = []

    for dst_nbr in sorted(nx_G.neighbors(dst)):

        if dst_nbr == src:

            unnormalized_probs.append(nx_G[dst][dst_nbr]['weight'] / p)

        elif nx_G.has_edge(dst_nbr, src):

            unnormalized_probs.append(nx_G[dst][dst_nbr]['weight'])

        else:

            unnormalized_probs.append(nx_G[dst][dst_nbr]['weight'] / q)

    norm_const = sum(unnormalized_probs)

    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)
def alias_draw(J, q):   

    K = len(J)

    kk = int(np.floor(np.random.rand() * K))

    

    if np.random.rand() < q[kk]:

        return kk

    else:

        return J[kk]

def node2vec_walk(nx_G, alias_nodes, alias_edges, walk_length, start_node):



    sampled_edges = defaultdict(set)

    walk = [start_node]

    

    while len(walk) < walk_length:

        cur = walk[-1]

        cur_nbrs = sorted(nx_G.neighbors(cur))

        if len(cur_nbrs) > 0:

            if len(walk) == 1:



                next = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]

                walk.append(next)

            else:

                prev = walk[-2]

                next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]

                walk.append(next)

        else:

            break



    return walk
write_random_walks = open('./random_walks_train.txt', 'w')

rfile = open('../input/deepcas-paper-dataset/cascade_train.txt', 'r')
for line in rfile:

        

    str_list, nx_G = parse_line(line)

    alias_nodes = {}

    for node in nx_G.nodes():

            unnormalized_probs = [nx_G[node][nbr]['weight'] for nbr in sorted(nx_G.neighbors(node))] # Sampling probabilities of neighbors are in proportion to neighbor weights.

            norm_const = sum(unnormalized_probs)

            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

            alias_nodes[node] = alias_setup(normalized_probs)  # A preprocess used to efficiently sample from multinomial distribution.



    alias_edges = {edge: get_alias_edge(nx_G, edge[0], edge[1]) for edge in nx_G.edges()}

    walks_per_graph = 200

    first_time = True

    while True:

        if first_time:

            first_time = False

            node_list,prob_list = extract_first_time(nx_G)

        else:

            node_list,prob_list = extract(nx_G)



        n_sample = min(len(node_list), walks_per_graph)

        if n_sample <= 0: break

        walks_per_graph -= n_sample

        sampled_nodes = np.random.choice(node_list, n_sample, replace=False, p=prob_list)



        

        walks = []

        walk_cnt = 0

        num_walks = len(sampled_nodes)

        for walk_iter in range(num_walks):

            for node in sampled_nodes:

                walks.append(node2vec_walk(nx_G, alias_nodes, alias_edges, walk_length=10, start_node=node))

                walk_cnt += 1

                if walk_cnt % 5000 == 0:  print("Current walks: ", walk_cnt)

                if walk_cnt >= num_walks: break

            if walk_cnt >= num_walks: break

            

        

        for walk in walks:

            str_list.append(' '.join(str(k) for k in walk))

    walk_string = '\t'.join(str_list)



    write_random_walks.write(walk_string + "\n")