from strawberryfields.apps import data, sample, subgraph, plot

import plotly

import networkx as nx



planted = data.Planted()

postselected = sample.postselect(planted, 16, 30)

pl_graph = nx.to_networkx_graph(planted.adj)

samples = sample.to_subgraphs(postselected, pl_graph)

print(len(samples))



sub = list(range(20, 30))

plot_graph = plot.graph(pl_graph, sub)

plotly.offline.plot(plot_graph, filename="planted.html")