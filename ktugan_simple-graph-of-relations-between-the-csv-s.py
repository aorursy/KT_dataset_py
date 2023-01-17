%matplotlib notebook

import os

import networkx as nx

import seaborn as sns

import matplotlib.pyplot as plt
# Read headers in each csv.

headers = {}

for csv in os.listdir('../input'):

    with open('../input/{}'.format(csv)) as f:

        headers[csv] = f.readline().rstrip().split(',')
# Make a graph out of it

g = nx.Graph()



for e, cols in headers.items():

	for c in cols:

		g.add_edge(e, c)
def plot_graph(g): # we reuse it later

    p = sns.color_palette("Paired", 4)[::2] # use nicer colors

    pos = nx.fruchterman_reingold_layout(g)

    colors = [p[0] if '.csv' in n else p[1] for n in g.nodes()]

    colors = [p[0] if '.csv' in n else p[1] for n in g.nodes()]

    

    plt.figure(figsize=(12,12), dpi=200)

    nx.draw_networkx_nodes(

        g,pos,

        nodelist=g.nodes(),

        node_color=colors,

        node_size=1000,

        alpha=0.8

    )

    nx.draw_networkx_edges(g, pos)

    nx.draw_networkx_labels(g, pos, font_size=9)



    plt.show()

    

plot_graph(g)
g2 = g.copy()

removable_nodes = [n for n, d in g2.degree_iter() if d == 1]

g2.remove_nodes_from(removable_nodes)



plot_graph(g2)