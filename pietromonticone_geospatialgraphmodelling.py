# INSTALL 

!pip install powerlaw

!pip install EoN

!pip install netwulf

!pip install progressbar
# IMPORT 



# Network Data Science 

import networkx as nx



# Data Wrangling

import numpy as np

import pickle



# Network Data Analysis 

import networkx as nx 



# Data Visualization

import seaborn as sns

import matplotlib.pyplot as plt 

import matplotlib.ticker as ticker

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.patheffects as path_effects

from mpl_toolkits.mplot3d.art3d import Line3DCollection

%pylab inline



# Geo-Spatial Data Analysis

import geopandas as gpd

import pandas as pd

import contextily

from shapely.geometry import Point



# Other Utilities

from operator import itemgetter

from collections import defaultdict



# Custom Modules

from shutil import copyfile

copyfile(src = "../input/epidemicmodelling/digital_epidemiology.py", dst = "../working/digital_epidemiology.py")

import digital_epidemiology as depi



# Reload Custom Modules

from importlib import reload

depi = reload(depi)
# Set geographic file path

regions = "../input/geospatial-italy/Reg01012020_WGS84.shp"

provinces = "../input/geospatial-italy/ProvCM01012020_WGS84.shp"

municipalities = "../input/geospatial-italy/Com01012020_WGS84.shp"



# Set demographic file path

regional_demographics = "../input/geospatial-italy/RegionalDemographics.csv"

provincial_demographics = "../input/geospatial-italy/ProvincialDemographics.csv"

municipal_demographics = "../input/geospatial-italy/MunicipalDemographics.csv"
# Store geographic data

regional_data = gpd.read_file(regions)

# Store demographic data

regional_demo_data = pd.read_csv(regional_demographics)



# Expand geographic data 

depi.expand(data = regional_data,          # geographic data

            level = "regional",            # geographic scale

            demo = regional_demo_data)     # demographic data

    

# Visualize geographic data 

regional_data.plot(figsize=(10, 10)).set_axis_off()
# Build the undirected graph

regional_graph = depi.build_graph(data = regional_data,   # geographic data

                                  level = "regional",     # geographic scale

                                  graph = nx.Graph())     # graph data



# Create nodal population list

regional_pop_list = [10000*regional_graph.pop[i] for i in list(regional_graph.nodes)] / sum([regional_graph.pop[i] for i in list(regional_graph.nodes)])
# Showcase connectivity

print("CONNECTIVITY:\n•",regional_graph.number_of_nodes(),"nodes;\n•",regional_graph.number_of_edges(),"edges.")



# Visualize graph 

fig=plt.figure(figsize=(15,8))

nx.draw_networkx_nodes(regional_graph, pos=regional_graph.pos,node_size=regional_pop_list)

nx.draw_networkx_edges(regional_graph, pos=regional_graph.pos, alpha=0.3) 

nx.draw_networkx_labels(regional_graph, pos=regional_graph.pos)

plt.axis('off')

plt.show()
# Save graphml  

nx.write_graphml(regional_graph, "../working/RegionalGraph.graphml")



# Save edgelist

nx.write_edgelist(regional_graph, "../working/RegionalGraph.csv",

                  delimiter=',',

                  data=False)
# Store geographic data

provincial_data = gpd.read_file(provinces)

# Store demographic data

provincial_demo_data = pd.read_csv(provincial_demographics)



# Expand geographic data 

depi.expand(data = provincial_data,        # geographic data

            level = "provincial",          # geographic scale

            demo = provincial_demo_data)   # demographic data

    

# Visualize geographic data 

provincial_data.plot(figsize=(10, 10)).set_axis_off()
# Build the undirected graph

provincial_graph = depi.build_graph(data = provincial_data,   # geographic data

                                  level = "provincial",       # geographic scale

                                  graph = nx.Graph())         # graph data



# Create nodal population list

provincial_pop_list = [10000*provincial_graph.pop[i] for i in list(provincial_graph.nodes)] / sum([provincial_graph.pop[i] for i in list(provincial_graph.nodes)])
# Showcase connectivity

print("CONNECTIVITY:\n•",provincial_graph.number_of_nodes(),"nodes;\n•",provincial_graph.number_of_edges(),"edges.")



# Visualize Graph

fig=plt.figure(figsize=(15,8))

nx.draw_networkx_nodes(provincial_graph, pos=provincial_graph.pos,node_size=provincial_pop_list)

nx.draw_networkx_edges(provincial_graph,pos=provincial_graph.pos, alpha = 0.5)

#nx.draw_networkx_labels(H, pos=H.pos)

plt.axis('off')

plt.show()
# Save graphml  

nx.write_graphml(provincial_graph, "../working/ProvincialGraph.graphml")



# Save edgelist

nx.write_edgelist(provincial_graph, "../working/ProvincialGraph.csv",

                  delimiter=',',

                  data=False)
# Store geographic data

municipal_data = gpd.read_file(municipalities)

# Store demographic data

municipal_demo_data = pd.read_csv(municipal_demographics)



# Expand geographic data 

depi.expand(data = municipal_data,        # geographic data

            level = "municipal",          # geographic scale

            demo = municipal_demo_data)   # demographic data

    

# Visualize geographic data 

municipal_data.plot(figsize=(10, 10)).set_axis_off()
# Build the undirected graph

municipal_graph = depi.build_graph(data = municipal_data,   # geographic data

                                  level = "municipal",      # geographic scale

                                  graph = nx.Graph())       # graph data



# Create nodal population list

municipal_pop_list = [10000*municipal_graph.pop[i] for i in list(municipal_graph.nodes)] / sum([municipal_graph.pop[i] for i in list(municipal_graph.nodes)])
# Showcase connectivity

print("CONNECTIVITY:\n•",municipal_graph.number_of_nodes(),"nodes;\n•",municipal_graph.number_of_edges(),"edges.")



# Visualize Graph

fig=plt.figure(figsize=(15,8))

nx.draw_networkx_nodes(municipal_graph, pos=municipal_graph.pos,node_size=municipal_pop_list)

nx.draw_networkx_edges(municipal_graph,pos=municipal_graph.pos, alpha = 0.5)

plt.axis('off')

plt.show()
# Save graphml  

nx.write_graphml(municipal_graph, "../working/MunicipalGraph.graphml")



# Save edgelist

nx.write_edgelist(municipal_graph, "../working/MunicipalGraph.csv",

                  delimiter=',',

                  data=False)
# Create list of colors

cols = ['steelblue', 

        'darksalmon', 

        'mediumseagreen']



# Create list of graphs

graphs = [municipal_graph, 

          provincial_graph, 

          regional_graph]



# Set figure size

width = 55

height = 55



# Plot multi-layered illustration

fig, ax = plt.subplots(1, 1, figsize=(width,height), dpi=300, subplot_kw={'projection':'3d'})



for gi, G in enumerate(graphs):

    pos = {}       # create empty position dictionary

    lines3d = []   # create empty list or 3D-lines

    

    # make x,y coordinates of all sub-figures compatible with that with max area (i.p. municipal graph)

    xs = list(list(zip(*list(graphs[0].pos.values())))[0])  

    ys = list(list(zip(*list(graphs[0].pos.values())))[1])

    

    xdiff = max(xs)-min(xs)

    ydiff = max(ys)-min(ys)

    

    xmin = min(xs)-xdiff*0.1 * (width/height)

    xmax = max(xs)+xdiff*0.1 * (width/height)

    ymin = min(ys)-ydiff*0.1

    ymax = max(ys)+ydiff*0.1

    

    if gi == 0: 

        pos = G.pos

        lines3d = [(list(pos[int(i)])+[gi],list(pos[int(j)])+[gi]) for i,j in G.edges()]

        node_size = 1

    elif gi == 1: 

        pos = G.pos

        lines3d = [(list(pos[i])+[gi],list(pos[j])+[gi]) for i,j in G.edges()]

        node_size = 1000

    elif gi == 2:

        pos = G.pos

        lines3d = [(list(pos[i])+[gi],list(pos[j])+[gi]) for i,j in G.edges()]

        node_size = 7000

    

    # node positions

    xs = list(list(zip(*list(pos.values())))[0])

    ys = list(list(zip(*list(pos.values())))[1])

    zs = [gi]*len(xs)   # set a common z-position of the nodes



    # node colors

    cs = [cols[gi]]*len(xs)

    

    # add intra-layer edges 

    line_collection = Line3DCollection(lines3d, zorder=gi, color=cols[gi], alpha=0.8)

    ax.add_collection3d(line_collection)



    # add nodes

    ax.scatter(xs, ys, zs, s=node_size ,c=cs, edgecolors='.3', marker='.', alpha=1, zorder=gi)

    

    # add a plane to designate the layer

    xdiff = max(xs)-min(xs)

    ydiff = max(ys)-min(ys)

    xx, yy = np.meshgrid([xmin, xmax],[ymin, ymax])

    zz = np.zeros(xx.shape)+gi

    ax.plot_surface(xx, yy, zz, color=cols[gi], alpha=0.1, zorder=gi)



ax.set_ylim(min(ys)-ydiff*0.1,max(ys)+ydiff*0.1)

ax.set_xlim(min(xs)-xdiff*0.1,max(xs)+xdiff*0.1)

ax.set_zlim(-0.1, 2.1)  # manually fine-tuned



ax.view_init(23, -80)   # select viewing angle

ax.dist=9               # set zoom

ax.set_axis_off()



#plt.savefig('../working/MultiLayerItaly.png',dpi=400,bbox_inches='tight')

plt.show()