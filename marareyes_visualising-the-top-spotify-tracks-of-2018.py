import pandas as pd 

import igraph as ig

import matplotlib.pyplot as plt



# load data

data = pd.read_csv("../input/top-spotify-tracks-of-2018/top2018.csv")

data.head()
G = ig.Graph()

typeseq = []

for index,row in data.iterrows():

    G.add_vertices(row['name'])

    typeseq.append("song")

    # create a node for artist only if it doesn't exist yet

    if row['artists'] not in G.vs['name']:

        G.add_vertices(row['artists']) 

        typeseq.append("artist")

    G.add_edge(row['name'],row['artists'])



# define danceability attribute

G.es["danceability"] = data["danceability"]

ig.summary(G)
G.vs["type"] = typeseq

nodesize = [i.degree()*5 for i in G.vs]     # vary node size based on degree

nodecolor = ["#fbb4ae" if t=="song" else "#ccebc5" for t in G.vs["type"]]     # different node colors for artists and songs

nodelabel = [i["name"] if i.degree() >= 3 else '' for i in G.vs]     # put labels for artists with 3 or more songs

layout = G.layout("fr")



# color edges by danceability

edgecolor = []

for i in G.es["danceability"]:

    pal = ig.RainbowPalette(n=8, alpha=i)

    edgecolor.append(pal.get(0))



ig.plot(G, layout = layout, 

        vertex_size=nodesize, 

        vertex_color=nodecolor,

        vertex_label = nodelabel,

        vertex_label_color="#1f78b4",

        edge_color = edgecolor,

        bbox=(450,450))