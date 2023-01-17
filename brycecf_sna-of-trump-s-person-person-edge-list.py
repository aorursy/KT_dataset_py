import matplotlib as mpl

import matplotlib.pyplot as plt

import networkx as nx

import numpy as np

import pandas as pd

import seaborn as sns
person_df = pd.read_csv('../input/Person_Person.csv',

                        encoding='ISO-8859-1')

person_G = nx.from_pandas_dataframe(person_df,

                                    source='Person A',

                                    target='Person B',

                                    edge_attr='Connection')

print(nx.info(person_G))

print('Density %f' % (nx.density(person_G)))
person_df[(person_df['Person A'] == 'DONALD J. TRUMP') | 

          (person_df['Person B'] == 'DONALD J. TRUMP')].info()
pos = nx.spring_layout(person_G) # Needs to created ahead of time for a consistent graph layout



def draw_graph(dataframe, metric):

    node_sizes = dataframe[metric].add(.5).multiply(50).tolist()

    nodes = dataframe.index.values.tolist()

    edges = nx.to_edgelist(person_G)

    metric_G = nx.Graph()

    metric_G.add_nodes_from(nodes)

    metric_G.add_edges_from(edges)

    labels = {}

    color_map = {}

    for node in nodes[:25]:

        labels[node] = node

    plt.figure(1, figsize=(20,20))

    plt.title('Trump\'s Person-Person Network (%s)' % (metric))

    nx.draw(metric_G,

            pos=pos,

            node_size=node_sizes, 

            node_color='#747587',

            with_labels=False)

    nx.draw_networkx_nodes(metric_G,

                           pos=pos,

                           nodelist=nodes[:25],

                           node_color='#873737',

                           nodesize=node_sizes)

    nx.draw_networkx_nodes(metric_G,

                           pos=pos,

                           nodelist=nodes[25:],

                           node_color='#747587',

                           nodesize=node_sizes)

    nx.draw_networkx_edges(metric_G,

                           pos=pos,

                           edgelist=edges,

                           arrows=False)

    tmp = nx.draw_networkx_labels(metric_G,

                                  pos=pos,

                                  labels=labels,

                                  font_size=16,

                                  font_color='#000000',

                                  font_weight='bold')
person_betweenness = pd.Series(nx.betweenness_centrality(person_G), name='Betweenness')

person_person_df = pd.Series.to_frame(person_betweenness)

person_person_df['Closeness'] = pd.Series(nx.closeness_centrality(person_G))

person_person_df['PageRank'] = pd.Series(nx.pagerank_scipy(person_G))

desc_betweenness = person_person_df.sort_values('Betweenness', ascending=False)

print('Top Highest Betweenness Centrality Persons')

desc_betweenness.head(25)
draw_graph(desc_betweenness, 'Betweenness')
desc_closeness = person_person_df.sort_values('Closeness', ascending=False)

print('Top Highest Closeness Centrality Persons')

desc_closeness.head(25)
draw_graph(desc_closeness, 'Closeness')
desc_pagerank = person_person_df.sort_values('PageRank', ascending=False)

print('Top Highest PageRank Persons')

desc_pagerank.head(25)
draw_graph(desc_pagerank, 'PageRank')
edited_G = person_G.copy()

edited_G.remove_node('DONALD J. TRUMP')

print('High-level Metrics in Graph w/ Trump')

print('='*10)

print(nx.info(person_G))

print('Density %f\n' % (nx.density(person_G)))

print('High-level Metrics in Graph w/o Trump')

print('='*10)

print(nx.info(edited_G))

print('Density %f\n' % (nx.density(edited_G)))

kushner_G = person_G.copy()

kushner_G.remove_node('JARED KUSHNER')

print('High-level Metrics in Graph w/o Jared Kushner')

print('='*10)

print(nx.info(kushner_G))

print('Density %f' % (nx.density(kushner_G)))