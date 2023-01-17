# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('/kaggle/input/vienna-subway-network/Vienna subway.csv', sep=';')

df.head()
stations = pd.unique(df[['Start', 'Stop']].values.flatten())

stations = sorted(stations)

station2id = {s:i for i, s in enumerate(stations)}

id2station = dict(enumerate(stations))
import networkx as nx

nx.__version__
G = nx.Graph()
for station_name, id in station2id.items():

    G.add_node(id, name=station_name)
for i, series in df.iterrows():

    u = station2id[series['Start']]

    v = station2id[series['Stop']]

    line = series['Line']

    color = series['Color']

    G.add_edge(u, v, line=line, color=color)
import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize=(20,15))



# calculate posisions in which the nodes are to be placed.

pos = nx.spring_layout(G, k=0.05, seed=2020)



# draw edges

unique_colors = pd.unique(list(nx.get_edge_attributes(G, 'color').values()))

for color in unique_colors:

    edgelist = [e for e, c in nx.get_edge_attributes(G, 'color').items() if c == color]

    nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=color, width=5)



# draw nodes

nx.draw_networkx_labels(G, pos, id2station, alpha=0.5)

nx.draw_networkx_nodes(G, pos, node_color='white', node_size=100)



plt.axis("off")

plt.show()
pr = nx.pagerank(G)
for i in range(10):

    print('[{}] pageranke score: {}'.format(i, pr[i]))
plt.figure(figsize=(20,15))



# draw edges

for color in unique_colors:

    edgelist = [e for e, c in nx.get_edge_attributes(G, 'color').items() if c == color]

    nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=color, width=5)



# draw nodes

node_size = [100000 * value for key, value, in pr.items()]

nx.draw_networkx_labels(G, pos, id2station, alpha=0.5)

nx.draw_networkx_nodes(G, pos, node_color='gray', alpha=0.8, node_size=node_size)



plt.axis("off")

plt.show()
!cd /tmp && git clone https://github.com/phanein/deepwalk.git && pip install ./deepwalk
from deepwalk import graph

import random

from gensim.models import Word2Vec



# parameter # (default parameter)

n_walks = 50 # 10

walk_length = 100 # 40

n_workers = 1

representation_size = 2

window_size = 10 # 5

seed = 2020



walks = graph.build_deepwalk_corpus(graph.from_networkx(G), num_paths=n_walks, path_length=walk_length, alpha=0, rand=random.Random(seed))

model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1, workers=n_workers)

model.wv.save_word2vec_format('/tmp/deepwalk.out')
deepwalk_out = pd.read_csv('/tmp/deepwalk.out', header=0, sep=' ', names=['x1', 'x2'])

embed_pos = {i: [row['x1'], row['x2']] for i, row in deepwalk_out.iterrows()}



deepwalk_out.head()
plt.figure(figsize=(16,12))



# draw edges

unique_colors = pd.unique(list(nx.get_edge_attributes(G, 'color').values()))

for color in unique_colors:

    edgelist = [e for e, c in nx.get_edge_attributes(G, 'color').items() if c == color]

    nx.draw_networkx_edges(G, embed_pos, edgelist=edgelist, edge_color=color, width=5)



# draw nodes

nx.draw_networkx_nodes(G, embed_pos, node_color='gray', alpha=0.8, node_size=200)



plt.axis("off")

plt.show()
! cd /tmp && git clone https://github.com/openjny/role2vec && pip install texttable
nx.write_edgelist(G, "/tmp/G.edgelist", data=False, delimiter=',')
! cd /tmp/role2vec && python src/main.py --dimensions 2 --walk-number 20 --window-size 10 --graph-input /tmp/G.edgelist --output /tmp/role2vec.out
role2vec_out = pd.read_csv('/tmp/role2vec.out', index_col='id')

embed_pos = {i: [row['x_0'], row['x_1']] for i, row in role2vec_out.iterrows()}

role2vec_out.head()
plt.figure(figsize=(16,12))



# draw edges

unique_colors = pd.unique(list(nx.get_edge_attributes(G, 'color').values()))

for color in unique_colors:

    edgelist = [e for e, c in nx.get_edge_attributes(G, 'color').items() if c == color]

    nx.draw_networkx_edges(G, embed_pos, edgelist=edgelist, edge_color=color, width=5)



# draw nodes

nx.draw_networkx_nodes(G, embed_pos, node_color='gray', alpha=0.8, node_size=200)



plt.axis("off")

plt.show()
!pip install dgl rdflib 
import dgl

dgl.__version__
def to_dgl(G):

    g = dgl.DGLGraph()

    g.add_nodes(len(G))

    edge_list = [(u,v) for u,v in nx.edges(G)]

    src, dst = tuple(zip(*edge_list))

    g.add_edges(src, dst)

    g.add_edges(dst, src)

    return g



g = to_dgl(G)



# You can revert this convertion with the `to_networkx` function.

# G = g.to_networkx().to_undirected()
import torch



onehot_features = torch.eye(len(G))

g.ndata['feat'] = onehot_features
# For example, the feature vector of the 4th node goes like: 

print(g.nodes[3].data['feat'])
import torch.nn as nn

import torch.nn.functional as F



# Define the message & reduce function

# NOTE: we ignore the GCN's normalization constant c_ij for this tutorial.

def gcn_message(edges):

    # The argument is a batch of edges.

    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.

    return {'msg' : edges.src['h']}



def gcn_reduce(nodes):

    # The argument is a batch of nodes.

    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.

    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}



# Define the GCNLayer module

class GCNLayer(nn.Module):

    def __init__(self, in_feats, out_feats):

        super(GCNLayer, self).__init__()

        self.linear = nn.Linear(in_feats, out_feats)



    def forward(self, g, inputs):

        # g is the graph and the inputs is the input node features

        # first set the node features

        g.ndata['h'] = inputs

        # trigger message passing on all edges

        g.send(g.edges(), gcn_message)

        # trigger aggregation at all nodes

        g.recv(g.nodes(), gcn_reduce)

        # get the result node features

        h = g.ndata.pop('h')

        # perform linear transformation

        return self.linear(h)

    

# Define a 2-layer GCN model

class BasicGCN(nn.Module):

    def __init__(self, in_feats, hidden_size, num_classes):

        super(BasicGCN, self).__init__()

        self.gcn1 = GCNLayer(in_feats, hidden_size)

        self.gcn2 = GCNLayer(hidden_size, num_classes)



    def forward(self, g, inputs):

        h = self.gcn1(g, inputs)

        h = torch.relu(h)

        h = self.gcn2(g, h)

        return h
plt.figure(figsize=(15,15))

unique_colors = pd.unique(list(nx.get_edge_attributes(G, 'color').values()))

for color in unique_colors:

    edgelist = [e for e, c in nx.get_edge_attributes(G, 'color').items() if c == color]

    nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=color, width=5)

nx.draw_networkx_labels(G, pos, {i:str(i) for s,i in station2id.items()}, font_size=18, alpha=0.8)

nx.draw_networkx_nodes(G, pos, node_color='white', node_size=400)

plt.axis("off")

plt.show()
unique_colors
from progressbar import progressbar 



# The first layer transforms input features of size of #nodes to a hidden size of 10.

# The second layer transforms the hidden layer and produces output features of

# size 5, corresponding to the five colors of the representative nodes.

net = BasicGCN(len(g.nodes), 10, 5)



inputs = torch.eye(len(g.nodes))

labeled_nodes = torch.tensor([60,47,38,20,77,35,76,80,31,69,78,61])  # only the instructor and the president nodes are labeled

labels = torch.tensor([0,0,0,1,1,1,2,2,3,3,4,4])  # their labels are different



# Options for learning

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

n_epochs = 100



all_logits = []

losses = []



for epoch in progressbar(range(n_epochs)):

    logits = net(g, inputs)

    # we save the logits for visualization later

    all_logits.append(logits.detach())

    logp = F.log_softmax(logits, 1)

    # we only compute loss for labeled nodes

    loss = F.nll_loss(logp[labeled_nodes], labels)

    losses.append(loss.item())



    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
def plot_loss(losses, n_epochs, **kwargs):

    fig, ax = plt.subplots(**kwargs)

    ax.plot(range(n_epochs), losses)

    ax.set_xlabel("epochs")

    ax.set_ylabel("loss")

    ax.set_title("Loss by epoch")

    plt.show()
plot_loss(losses, n_epochs)
epoch = -10

node_id = 41



print('node #{} in epoch #{}'.format(node_id, epoch))

print('logits:', all_logits[epoch][node_id].numpy())

print('the most possible class:', unique_colors[all_logits[epoch][node_id].argmax().numpy()])
predictions = [unique_colors[x] for x in all_logits[-1].argmax(axis=-1)]
def plot_predictions(pos, predictions):

    plt.figure(figsize=(16,12))



    nx.draw_networkx_nodes(G, pos, node_color=predictions, alpha=0.5, node_size=300)

    nx.draw_networkx_nodes(G, pos, nodelist=labeled_nodes.tolist(), node_color=[unique_colors[l] for l in labels], alpha=0.9, node_size=1000)



    for color in unique_colors:

        edgelist = [e for e, c in nx.get_edge_attributes(G, 'color').items() if c == color]

        nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=color, width=2)



        colored_labels = {i:i for i,c in dict(enumerate(predictions)).items() if c == color}

        nx.draw_networkx_labels(G, pos, colored_labels, font_size=10, font_color=color)



    plt.axis("off")

    plt.show()



plot_predictions(pos, predictions)
import torch

import torch.nn as nn

from dgl.nn.pytorch import GraphConv



class GCN(nn.Module):

    def __init__(self,

                 g,

                 in_feats,

                 n_hidden,

                 n_classes,

                 n_layers,

                 activation,

                 dropout):

        super(GCN, self).__init__()

        self.g = g

        self.dropout = nn.Dropout(p=dropout)

        

        self.layers = nn.ModuleList()

        # input layer

        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))

        # hidden layers

        for i in range(n_layers - 1):

            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))

        # output layer

        self.layers.append(GraphConv(n_hidden, n_classes))



    def forward(self, features):

        h = features

        for i, layer in enumerate(self.layers):

            if i != 0:

                h = self.dropout(h)

            h = layer(self.g, h)

        # output logits

        return h
# cuda option

cuda = False



# normalization term

degs = g.in_degrees().float()

norm = torch.pow(degs, -0.5)

norm[torch.isinf(norm)] = 0

if cuda:

    norm = norm.cuda()

g.ndata['norm'] = norm.unsqueeze(1)



# hyper parameters

n_epochs = 100

n_layers = 3

n_hidden = 16

dropout = 0.3

weight_decay = 5e-4

lr=0.01



# feature and target class

features  = torch.eye(len(g.nodes))

in_feats = features.shape[1]

n_classes = len(unique_colors)



# supervise data

labeled_nodes = torch.tensor([60,47,20,77,76,80,31,69,78,61])

labels = torch.tensor([0,0,1,1,2,2,3,3,4,4])



model = GCN(g,

            in_feats,

            n_hidden,

            n_classes,

            n_layers,

            F.relu,

            dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



all_logits = []

losses = []



n_epochs = 200

for epoch in progressbar(range(n_epochs)):

    model.train()

    logits = model(features)

#     all_logits.append(logits.detach())



    logp = F.log_softmax(logits, 1)

    loss = F.nll_loss(logp[labeled_nodes], labels)

    losses.append(loss.item())



    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
plot_loss(losses, n_epochs)
predictions = [unique_colors[x] for x in logits.argmax(axis=-1)]

plot_predictions(pos, predictions)
import torch

import torch.nn as nn

import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv

from dgl.nn.pytorch import SumPooling, MaxPooling



# https://github.com/pengchenghu428/GraphTools/blob/master/experiment/pytorch/gcn.py

class AdvancedGCN(nn.Module):

    def __init__(self,

                 g,

                 in_feats,

                 n_output=2,

                 n_hidden_gcn=[256, 256, 256],

                 n_hidden_mlp=[32],

                 n_dense_hidden=128,

                 activation=F.leaky_relu,

                 dropout=0.2):

        super(AdvancedGCN, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.activation = activation

        self.n_dense_hidden = n_dense_hidden

        self.n_hidden_gcn = n_hidden_gcn

        self.g = g

        

        # GCN layer

        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(GraphConv(in_feats, n_hidden_gcn[0], activation=activation))

        for i in range(len(n_hidden_gcn)-1):

            self.gcn_layers.append(GraphConv(n_hidden_gcn[i], n_hidden_gcn[i+1], activation=activation))

        

        # MLP layer

        n_hidden_mlp.append(n_output)

        self.mlp_layers = nn.ModuleList()

        self.mlp_layers.append(nn.Linear(n_hidden_gcn[-1], n_hidden_mlp[0]))

        for i in range(len(n_hidden_mlp)-1):

            self.mlp_layers.append(nn.Linear(n_hidden_mlp[i], n_hidden_mlp[i+1]))



    def forward(self, features):

        self.h = []

        h = features

        

        # GCN

        for i, gcn_layer in enumerate(self.gcn_layers):

            if i != 0:

                h = self.dropout(h)

            h = gcn_layer(self.g, h)

            self.h.append(h)

            

        # MLP

        for i, mlp_layer in enumerate(self.mlp_layers):

            if i != 0:

                h = self.dropout(h)

            h = mlp_layer(h)

            h = F.relu(h)

            self.h.append(h)

        

        return h

    

    def _get_flatten_size(self, x):

        size = x.size()[1:]

        num_features = 1

        for s in size:

            num_features *= s

        return num_features



model = AdvancedGCN(g,

                    in_feats,

                    n_hidden_gcn=[16, 16],

                    n_hidden_mlp=[8],

                    n_output=n_classes,

                    dropout=0.2)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



all_logits = []

losses = []



n_epochs = 100

for epoch in progressbar(range(n_epochs)):

    model.train()

    logits = model(features)

    

    logp = F.log_softmax(logits, 1)

    loss = F.nll_loss(logp[labeled_nodes], labels)

    losses.append(loss.item())



    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
plot_loss(losses, n_epochs)
predictions = [unique_colors[x] for x in logits.argmax(axis=-1)]

plot_predictions(pos, predictions)
for h in model.h:

    print(h.shape)
for i, h in enumerate(model.h):

    if h.shape[1] == 2:

        embed_pos = {i: [node[0], node[1]] for i, node in enumerate(h.detach())}

        plot_predictions(embed_pos, predictions)

        break
from sklearn.manifold import TSNE



embedded = TSNE(n_components=2).fit_transform(model.h[-1].detach())

embed_pos = {i: [node[0], node[1]] for i, node in enumerate(embedded)}

plot_predictions(embed_pos, predictions)