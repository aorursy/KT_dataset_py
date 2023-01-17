import random

from pprint import pprint

import networkx as nx

import matplotlib.pyplot as plt

import numpy as np

from math import inf, nan

import json

import pickle
data = open("../input/data").read()

entries = [x for x in data.split(";;") if x != ""]    # Divide data by date and remove empty entries

entry = entries[-1]                                   # The last entry, will be used here for demonstration
def split_entry(entry):

    dt = entry.split(":")[0]    # The entry's timestamp

    users = {}                  # Users' contacts {uid1: [uid11, uid12, uid13, ...], uid2: [uid21, uid22, ...], ...}

    

    # Split entry into single users and save their ids and their contacts' ids into `users`.

    for user in entry.split(":")[1].split(";"):

        users[user.split(".")[0]] = [x for x in user.split(".")[1][1:-1].split(", ") if x != ""]

    

    return dt, users
dt, users = split_entry(entry)

uids = list(users.keys())

users_test = random.sample([u for u in users if users[u]], 10)    # Pick 10 random connected users for demonstration

users_test_idx = [uids.index(u) for u in users_test]              # Also save indices of `users_test`

pprint(dict(random.sample(users.items(), 1)))                     # Print 1 random sample from `users`
def visualize_network(network, special=set(), labels=False, small=False):

    G = nx.DiGraph()    # Directed Graph -> the thicker parts of lines are "arrow heads", {a: [b]} => (a -> b)

    

    for n in network.items():

        for c in n[1]:

            G.add_edge(n[0], c)    # Because only edges are added to `G`, isolated nodes (=users) won't be shown

    

    node_colors = ['r' if x in special else 'b' for x in G.nodes()]

    

    plt.rc('figure', figsize=((6, 4) if small else (15, 10)))

    nx.draw_networkx(G, with_labels=labels, node_color=node_colors, node_size=150, width=0.5)

    plt.show()



print(f"Network of ePotato users on {dt}:")

visualize_network(users, set(users_test))
def separation(network):

    S = np.full((len(network), len(network)), inf)    # Initialize S to infinite separations

    nodes = list(network.keys())

    

    # Loop over all nodes, each time building up that node's row in the matrix.

    for node in network:

        levels = [[node]]    # Node X in levels[n] => Separation `node` -> user X: n°

        known = [node]       # Every node only has to be sorted into levels once (the lowest possible level)

        

        # Fill up `levels` until there are no new nodes to sort

        for level in levels:

            for node_ in level:

                for node__ in network[node_]:

                    if node__ not in known:

                        if level is levels[-1]:

                            levels.append([])

                        known.append(node__)

                        levels[-1].append(node__)

        

        # Write out level-numbers as columns in node's row in S

        for i in range(len(levels)):

            for node_ in levels[i]:

                S[nodes.index(node), nodes.index(node_)] = i

    

    return S
def visualize_matrix(mat):

    plt.rc('figure', figsize=(15.0, 10.0))

    plt.imshow(mat, cmap="jet")

    plt.colorbar()

    plt.show()



S = separation(users)

S_ = S[users_test_idx, :][:, users_test_idx]

print(f"Directed Separation Matrix (Extract):\n\n{S_}\n\n\nRepresentation:")

visualize_matrix(S_)
def misdirect(network):

    # Remove all directed edges from network

    net = {k: [c for c in network[k] if c in network and k in network[c]] for k in network}

    return net
S_tight = separation(misdirect(users))

S_ = S_tight[users_test_idx, :][:, users_test_idx]

print(f"Undirected (Misdirected) Separation Matrix (Extract):\n\n{S_}\n\n\nRepresentation:")

visualize_matrix(S_)
def loosen(network):

    net = network.copy()

    for n, cs in list(net.items()):

        for c in cs:

            net[c] = list(set(net.get(c, [])) | {n})    # Add node to all connected nodes

    return net
S_loose = separation(loosen(users))

S_ = S_loose[users_test_idx, :][:, users_test_idx]

print(f"Undirected (Loose) Separation Matrix (Extract):\n\n{S_}\n\n\nRepresentation:")

visualize_matrix(S_)
def simplify(network, resistors, a, b):

    net = network.copy()

    rs = resistors.copy()

    

    again = True

    while again:

        again = False      # Will be set to True if simplification was found

        active = {a, b}    # Active nodes: Those that are important for the resistance

        

        # Active nodes need to have incoming and outgoing connections.

        for ns in net.values():

            active |= {n for n in ns if n in net and net[n]}

        

        net_ = net.copy()    # Backup to check for changes

        

        # Remove all non-active nodes from the network

        net = {k: [c for c in net[k] if c in active and c != a] for k in active if k != b}

        rs = {k: rs[k] for k in rs if set(k) <= active and k[0] != b and k[1] != a}

        

        # Network has changed -> repeat from the top

        if net != net_:

            again = True

            continue

        

        # Loop over nodes with only one outgoing connection -> check for incoming connections

        for n, cs in [(n, cs) for n, cs in net.items() if len(cs) == 1 and n != a]:

            # Incoming nodes that are different from the outgoing node

            incoming = [x[0] for x in net.items() if n in x[1] and x[0] != cs[0]]



            # No incoming connections found -> delete semi-isolated node and repeat from the top

            if not incoming and n != b:

                del net[n]

                again = True

                break



            # One incoming connection -> redundand node, can be simplified as series connection

            if len(incoming) == 1:

                i = incoming[0]; o = cs[0]

                R = rs[i, n] + rs[n, o]    # Resulting resistor of series circuit



                # There is no existing edge -> Just simplify the series circuit, add new edge to network

                if (i, o) not in rs:

                    rs[i, o] = R

                    net[i].append(o)



                # There is an existing edge -> Simplify series and parallel circuits

                else:

                    rs[i, o] = R * rs[i, o] / (R + rs[i, o])



                # Delete redundand node from network and repeat from the top

                del net[n]

                again = True

                break

                    

    return net, rs
def build_resistors(network):

    resistors = {}

    # Loop over edges, initialize every resistor to 1 Ohm

    for node in network.items():

        for node_ in node[1]:

            resistors[node[0], node_] = 1    # `resistors`: {(node1, node2): R}

    return resistors
def star_mesh_transform(network, resistors, a, b):

    net = network.copy()

    rs = resistors.copy()

    

    # `n0`: Middle node of star, `cs`: Edges of star

    n0, cs = next((n0, cs) for n0, cs in net.items() if n0 not in {a, b})

    Rs = {}

    

    # Calculate new resistors and reconnect network

    for n in list(cs):

        for m in [c for c in cs if c != n]:

            # Calculate R_nm with star-mesh formula

            R = rs[n, n0] * rs[m, n0] * sum([1 / R for R in [R for k, R in rs.items() if k[0] == n0]])



            # Add new edge (double-sided)

            net[n] = list(set(net.get(n, [])) | {m})

            net[m] = list(set(net.get(m, [])) | {n})



            # Condense resistors

            rs[n, m] = rs[m, n] = R * rs[n, m] / (R + rs[n, m]) if (n, m) in rs else R



        # Remove star-edges

        net[n].remove(n0)

        net[n0].remove(n)



    # Remove old resistors

    rs = undirect(rs, net)

        

    return net, rs
def undirect(resistors, network):

    rs = {}

    # Add all resistors (double-sided) of which the edges are found in the network

    for k, R in [(k, R) for k, R in list(resistors.items()) if k[1] in network[k[0]]]:

        rs[k[0], k[1]] = rs[k[1], k[0]] = R

    return rs
def resistance(network, a, b, resistors=None, iterations=False, visualize=False):

    # Check if a = b

    if a == b:

        return (0, 0) if iterations else 0

    

    net = network.copy()

    

    # If resistors are not predefined, initialize them all to 1 Ohm (directed)

    if not resistors:

        rs = build_resistors(network)

    else:

        rs = resistors.copy()

    

    net_ = net.copy()    # Backup to check for changes

    i = 0                # Iterations counter

    

    while True:

        # Simplification

        net, rs = simplify(net, rs, a, b)

        

        if visualize:

            print("Simplification (Directed):")

            visualize_network(net, {a, b}, small=True, labels=True)

        

        # No changes or single connection: finished

        if net == net_ or set(net.keys()) <= {a, b}:

            break

            

        net_ = net.copy()    # Backup to check for changes

        

        # Star-Mesh Transform

        net = loosen(net)

        rs = undirect(rs, net)

        net, rs = star_mesh_transform(net, rs, a, b)

        

        i += 1

            

        if visualize:

            print(f"Star-Mesh Transform (Loosened Network) #{i}:")

            visualize_network(net, {a, b}, small=True, labels=True)

    

    R = nan    # Initialize R to NaN for when it can not be computed using star-mesh transform and simplification

    if len(rs) == 1:

        R = list(rs.values())[0]

    elif len(rs) == 0:

        R = inf

    

    return (R, i) if iterations else R
net = {1: [2, 3], 2: [3, 1, 5, 7], 3: [4, 2], 4: [3, 1, 5, 6], 5: [2, 6], 6: [2, 4], 7: [3]}

a = 1; b = 6



print("Network to be analyzed:")

visualize_network(net, {a, b}, labels=True)



print("R = %8.6fΩ (%d iterations) <- Directed Resistance 1 -> 6" % resistance(net, a, b, iterations=True, visualize=True))

print("R = %8.6fΩ (%d iterations) <- Directed Resistance 6 -> 1" % resistance(net, b, a, iterations=True))

print("R = %8.6fΩ (%d iterations) <- Misdirected Resistance" % resistance(misdirect(net), a, b, iterations=True))

print("R = %8.6fΩ (%d iterations) <- Loose Resistance" % resistance(loosen(net), a, b, iterations=True))
def resistance_distance(network):

    R = np.empty((len(network), len(network)))    # No intialization necessary, `resistance` always returns a value

    nodes = list(network.keys())

    

    for a in network:

        for b in network:

            R[nodes.index(a), nodes.index(b)] = resistance(network, a, b)

    

    return R
R = resistance_distance(users)

R_ = R[users_test_idx, :][:, users_test_idx]

print(f"Resistance Distance Matrix (Directed, Extract):\n\n{R_}\n\n\nRepresentation:")

visualize_matrix(R_)
R_tight = resistance_distance(misdirect(users))

R_ = R_tight[users_test_idx, :][:, users_test_idx]

print(f"Resistance Distance Matrix (Misdirected, Extract):\n\n{R_}\n\n\nRepresentation:")

visualize_matrix(R_)
R_loose = resistance_distance(loosen(users))

R_ = R_loose[users_test_idx, :][:, users_test_idx]

print(f"Resistance Distance Matrix (Loose, Extract):\n\n{R_}\n\n\nRepresentation:")

visualize_matrix(R_)
def propagate(network, a):

    net = {a: network[a]}      # The subnet to be built up

    history = set()            # Nodes already seen

    cache = set(network[a])    # Nodes to be crawled

    

    while cache:

        for node in cache.copy():

            net[node] = list(set(net.get(node, []) + network[node]))    # Add node to subnet

            cache |= {n for n in net[node] if n not in history}         # Add all newly accessible nodes to cache

            

            # We're done with this node

            history.add(node)

            cache.remove(node)

    

    return net
def subnets(network):

    nets = []        # The subnets to be found

    nodes = set()    # Already processed nodes

    

    for node in network:

        if node not in nodes:

            subnet = propagate(network, node)

            nets.append(subnet)                  # Add the subnet reachable from `node`

            nodes.update(subnet.keys())          # A node can only be in one subnet -> ignore all nodes in `subnet`

    

    return nets
nets = subnets(loosen(users))

largest = 0, 0

connected = 0

isolated = 0



for i, net in enumerate(nets):

    if len(net) > largest[1]:

        largest = i, len(net)

    if len(net) == 1:

        isolated += 1



cluster = nets[largest[0]]

percent = largest[1] / len(users) * 100



S_ = separation(cluster)

s = S_.sum() / (S_.size - len(S_))    # Average separation, remove 0° separations from calculation

        

R_ = resistance_distance(cluster)

r = R_.sum() / (R_.size - len(R_))    # Average resistance, remove 0° separations from calculation



print(f"""

There are {len(nets)} subnets in the network, {isolated} of which are isolated (population: 1 user). 

The biggest subnet has {largest[1]} members, making up {percent}% of the whole network. 

The average separation S of this largest cluster and its average resistance distance R are:

S = {s}

R = {r}Ω.

""")