import numpy as np

import pandas as pd

from google.cloud import bigquery



client = bigquery.Client()

query = """

SELECT from_address, 

    to_address

FROM `bigquery-public-data.crypto_ethereum_classic.transactions` 

GROUP BY from_address, to_address

ORDER BY from_address ASC   

    

"""

df = client.query(query).to_dataframe()

df.describe()
df = df.dropna()

df.describe()
!conda install --yes --override-channels -c ostrokach-forge -c pkgw-forge -c conda-forge gtk3 pygobject graph-tool
import graph_tool.all as gt



g = gt.Graph(directed=True)



# populate the graph with vertices and store their references in a dict that

# uniquely maps each address to a vertex

vertex_map = set(pd.concat([df['from_address'], df['to_address']], axis=0))

new_vertices = g.add_vertex(len(vertex_map))

vertex_map = dict(zip(vertex_map, new_vertices))



#add edges

def edge_map(e):

    return (vertex_map[e[0]], vertex_map[e[1]])

edge_list = map(edge_map, df[['from_address', 'to_address']].itertuples(index=False, name=None))

g.add_edge_list(edge_list)
comp, hist = gt.label_components(g, directed=False) #outputs a histogram of the # of vertices belonging to each component

print("The graph has", len(hist), "components")

print("the largest component has", max(hist), "vertices, %{:.2f} of the total.".format(100*max(hist)/len(g.get_vertices())))

print("the 2nd and 3rd largest components have {:d} and {:d} vertices respectively".format(sorted(hist, reverse=True)[1], sorted(hist, reverse=True)[2]))
import random

from tqdm import tqdm

from multiprocessing import Pool



sample_size = 10000



#get those vertices that belong to the largest component

comp = gt.label_largest_component(g, directed=False)

vertices_subset = [i for i, x in enumerate(comp) if x]



#randomly select vertex pairs

source_samples = random.choices(vertices_subset, k=sample_size)

target_samples = random.choices(vertices_subset, k=sample_size)

sample_pairs = zip(source_samples, target_samples)



def get_shortest_distance(pair):

    return gt.shortest_distance(g, pair[0], pair[1], directed=False)

pool = Pool(4)

distance_list = list(tqdm(pool.imap(get_shortest_distance, sample_pairs), total=sample_size))

pool.close()

pool.join()
import matplotlib.pyplot as plt



plt.figure(figsize=(10,5))

# plt.yscale("log")

# plt.xscale("log")

plt.title('Histogram of Minimum Path Length Between Vertex Pairs')

plt.xlabel('Shortest Path Length')

plt.ylabel('Count')



bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf]

hist, _ = np.histogram(distance_list, bins)



plt.xticks(range(len(bins)), bins[:-2] + ['{}+'.format(bins[-2])])

plt.bar(range(len(hist)), hist, width=1)

plt.show()
import datetime

start_time = datetime.datetime.now()



clustering = gt.global_clustering(g)

print("computation time:", datetime.datetime.now() - start_time)

print('The clustering coeficient of the transaction network is:', clustering[0])
import matplotlib.pyplot as plt



degrees = g.get_out_degrees(g.get_vertices())

plt.figure(figsize=(15,5))

plt.yscale("log")

plt.xscale("log")

plt.xlabel('Vertex Degree')

plt.ylabel('Count')



plt.hist(degrees, max(degrees))



plt.show()
comp, hist = gt.label_components(g, directed=True) # label strongly connected components

condensed_g, _, vcount, _, _, _ = gt.condensation_graph(g, comp) # generate a new graph from component labels



print("# of vertices in the original graph:", len(g.get_vertices()))

print("# of strongly connected components in the original graph:", len(hist))

print("# of vertices in the condensed graph:", len(condensed_g.get_vertices()))
import datetime



start_time = datetime.datetime.now()

pos = gt.sfdp_layout(condensed_g, epsilon=2.0, pos=None, verbose=True)

print("computation time:", datetime.datetime.now() - start_time)

gt.graph_draw(condensed_g, pos=pos, output_size=(5000, 5000),vertex_size=1.0,  edge_pen_width=0.1)