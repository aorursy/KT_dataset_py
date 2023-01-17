!apt-get install -y graphviz libgraphviz-dev libcgraph6
!pip install git+https://github.com/danielegrattarola/spektral
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from skimage.util import montage

from IPython.display import Image, display, SVG, clear_output, HTML

plt.rcParams["figure.figsize"] = (6, 6)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'gray' # grayscale looks better

import networkx as nx

def draw_graph_mpl(g, pos=None, ax=None, layout_func=nx.drawing.layout.kamada_kawai_layout, draw_labels=True):

    if ax is None:

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    else:

        fig = None

    if pos is None:

        pos = layout_func(g)

    node_color = []

    node_labels = {}

    shift_pos = {}

    for k in g:

        node_color.append(g.nodes[k].get('color', 'green'))

        node_labels[k] = g.nodes[k].get('label', k)

        shift_pos[k] = [pos[k][0], pos[k][1]]

    

    edge_color = []

    edge_width = []

    for e in g.edges():

        edge_color.append(g.edges[e].get('color', 'black'))

        edge_width.append(g.edges[e].get('width', 0.5))

    nx.draw_networkx_edges(g, pos, font_weight='bold', edge_color=edge_color, width=edge_width, alpha=0.5, ax=ax)

    nx.draw_networkx_nodes(g, pos, node_color=node_color, node_shape='p', node_size=300, alpha=0.75, ax=ax)

    if draw_labels:

        nx.draw_networkx_labels(g, shift_pos, labels=node_labels, arrows=True, ax=ax)

    ax.autoscale()

    return fig, ax, pos
from keras import Input, Model

from keras.callbacks import EarlyStopping

from keras.layers import Dense, Flatten

from keras.optimizers import Adam

from keras.regularizers import l2



from spektral.datasets import mnist

from spektral.layers import GraphConv

from spektral.layers.ops import sp_matrix_to_sp_tensor

from spektral.utils import normalized_laplacian
# Parameters

l2_reg = 5e-4         # Regularization rate for l2

learning_rate = 1e-3  # Learning rate for SGD

batch_size = 32       # Batch size

epochs = 5        # Number of training epochs

es_patience = 200     # Patience fot early stopping
# Load data

X_train, y_train, X_val, y_val, X_test, y_test, adj = mnist.load_data()

X_train, X_val, X_test = X_train[..., None], X_val[..., None], X_test[..., None]

N = X_train.shape[-2]      # Number of nodes in the graphs

F = X_train.shape[-1]      # Node features dimensionality

n_out = 10  # Dimension of the target



print(X_train.shape, 'model input')
print(adj.shape, 'adjacency matrix')

plt.matshow(adj.todense())
xx, yy = np.meshgrid(np.arange(28), np.arange(28))

node_id = ['X:{:02d}_Y:{:02d}'.format(x, y) for x, y in zip(xx.ravel(), yy.ravel())]
print(node_id[300], 'is connected to')

for row, col in zip(*adj[300].nonzero()):

    print(col, '->', node_id[col])
G = nx.from_scipy_sparse_matrix(adj[:10, :10])

for k, pix_val in zip(G.nodes, X_train[0]):

    G.nodes[k]['label'] = node_id[k]

draw_graph_mpl(G);
MAX_NODE = 28*12

G = nx.from_scipy_sparse_matrix(adj[:MAX_NODE, :MAX_NODE])

for k, pix_val in zip(G.nodes, X_train[0]):

    G.nodes[k]['label'] = node_id[k]

    G.nodes[k]['color'] = 'red' if pix_val>0.5 else 'green'

draw_graph_mpl(G);
G = nx.from_scipy_sparse_matrix(adj)

for k, pix_val in zip(G.nodes, X_train[0]):

    G.nodes[k]['label'] = ''

    G.nodes[k]['color'] = 'red' if pix_val>0.5 else 'green'

draw_graph_mpl(G, pos=np.stack([xx.ravel(), yy.ravel()], -1));
fltr = normalized_laplacian(adj)



# Model definition

X_in = Input(shape=(N, F))

# Pass A as a fixed tensor, otherwise Keras will complain about inputs of

# different rank.

A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))



graph_conv_1 = GraphConv(32,

                       activation='elu',

                       kernel_regularizer=l2(l2_reg),

                       use_bias=True)([X_in, A_in])

graph_conv_2 = GraphConv(32,

                       activation='elu',

                       kernel_regularizer=l2(l2_reg),

                       use_bias=True)([graph_conv_1, A_in])

flatten = Flatten()(graph_conv_2)

fc = Dense(512, activation='relu')(flatten)

output = Dense(n_out, activation='softmax')(fc)



# Build model

model = Model(inputs=[X_in, A_in], outputs=output)

optimizer = Adam(lr=learning_rate)

model.compile(optimizer=optimizer,

              loss='sparse_categorical_crossentropy',

              metrics=['acc'])

model.summary()
# Train model

validation_data = (X_val, y_val)

model.fit(X_train,

          y_train,

          batch_size=batch_size,

          validation_data=validation_data,

          epochs=epochs,

          callbacks=[

              EarlyStopping(patience=es_patience, restore_best_weights=True)

          ])
# Evaluate model

print('Evaluating model.')

eval_results = model.evaluate(X_test,

                              y_test,

                              batch_size=batch_size)

print('Test loss: {}\n'

      'Test acc: {}'.format(*eval_results))
W, b = model.layers[2].get_weights()

print(W.shape, b.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.bar(np.arange(W.shape[1]), W[0])

ax2.bar(np.arange(W.shape[1]), b)
W, b = model.layers[3].get_weights()

print(W.shape, b.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.imshow(W, vmin=-1, vmax=1, cmap='RdBu')

ax2.bar(np.arange(W.shape[1]), b)
i_model = Model(inputs=[X_in, A_in], outputs=[graph_conv_1, graph_conv_2])
gc1_out, gc2_out = i_model.predict(X_test[:32])
fig, m_axs = plt.subplots(4, 3, figsize=(20, 15))

for i, (ax1, ax2, ax3) in enumerate(m_axs):

    ax1.imshow(X_test[i].reshape((28, 28)))

    gc_stack = gc1_out[i].reshape((28, 28, -1)).swapaxes(0, 2).swapaxes(1, 2)

    ax2.imshow(montage(gc_stack), vmin=-0.5, vmax=0.5, cmap='RdBu')

    ax2.set_title(i_model.output_names[0])

    gc_stack = gc2_out[i].reshape((28, 28, -1)).swapaxes(0, 2).swapaxes(1, 2)

    ax3.imshow(montage(gc_stack), vmin=-0.5, vmax=0.5, cmap='RdBu')

    ax3.set_title(i_model.output_names[1])