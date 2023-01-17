!apt-get install -y graphviz libgraphviz-dev libcgraph6
!pip install -qq tensorflow-gpu==1.15.0

!pip install -qq git+https://github.com/danielegrattarola/spektral
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
from keras import Input, Model

from keras.callbacks import EarlyStopping

from keras.layers import Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.regularizers import l2



from spektral.datasets import mnist

from spektral.layers import GraphConv, GraphConvSkip, GraphAttention, GlobalAttentionPool, TopKPool, MinCutPool, DiffPool

from spektral.layers.ops import sp_matrix_to_sp_tensor

from spektral.utils import normalized_laplacian, normalized_adjacency
import networkx as nx

def draw_graph_mpl(g, pos=None, ax=None, layout_func=nx.drawing.layout.kamada_kawai_layout):

    if ax is None:

        fig, ax = plt.subplots(1, 1, figsize=(25, 25))

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

    

    nx.draw_networkx_edges(g, pos, font_weight='bold', ax=ax)

    nx.draw_networkx_nodes(g, pos, node_color=node_color, node_shape='p', node_size=300, alpha=0.75)

    nx.draw_networkx_labels(g, shift_pos, labels=node_labels, ax=ax, arrows=True)

    ax.autoscale()

    return fig, ax, pos
# Parameters

l2_reg = 5e-4         # Regularization rate for l2

learning_rate = 1e-3  # Learning rate for SGD

batch_size = 32       # Batch size

epochs = 10000        # Number of training epochs

es_patience = 15     # Patience fot early stopping
# Load data

X_train, y_train, X_val, y_val, X_test, y_test, adj = mnist.load_data()

X_train, X_val, X_test = X_train[..., None], X_val[..., None], X_test[..., None]

N = X_train.shape[-2]      # Number of nodes in the graphs

F = X_train.shape[-1]      # Node features dimensionality

n_out = 10  # Dimension of the target

fltr = normalized_laplacian(adj)

norm_adj = normalized_adjacency(adj)

print(X_train.shape, 'model input')
print(adj.shape, 'adjacency matrix')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

plot_adj_args = dict(cmap='RdBu', vmin=-0.5, vmax=0.5)

ax1.matshow(adj.todense(), **plot_adj_args)

ax1.set_title('Adjacency')

ax2.matshow(norm_adj.todense(), **plot_adj_args)

ax2.set_title('Normalized')

ax3.matshow(fltr.todense(),**plot_adj_args)

ax3.set_title('Laplacian Adjacency')
xx, yy = np.meshgrid(np.arange(28), np.arange(28))

node_id = ['X:{:02d}_Y:{:02d}'.format(x, y) for x, y in zip(xx.ravel(), yy.ravel())]
print(node_id[300], 'is connected to')

for row, col in zip(*adj[300].nonzero()):

    print(col, '->', node_id[col])
G = nx.from_scipy_sparse_matrix(adj[:10, :10])

for k in G.nodes:

    G.nodes[k]['label'] = node_id[k]

draw_graph_mpl(G);
MAX_NODE = 28*5

G = nx.from_scipy_sparse_matrix(adj[:MAX_NODE, :MAX_NODE])

for k in G.nodes:

    G.nodes[k]['label'] = node_id[k]

draw_graph_mpl(G);
G = nx.from_scipy_sparse_matrix(adj)

draw_graph_mpl(G, pos=np.stack([xx.ravel(), yy.ravel()], -1));
from keras import backend as K

K.clear_session()
# Model definition



X_in = Input(shape=(N, F))

# Pass A as a fixed tensor, otherwise Keras will complain about inputs of

# different rank.

A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr), name='LaplacianAdjacencyMatrix')



graph_conv_1 = GraphConv(32,

                       activation='elu',

                       kernel_regularizer=l2(l2_reg),

                       use_bias=True)([X_in, A_in])



graph_conv_2 = GraphConv(64,

                       activation='elu',

                       kernel_regularizer=l2(l2_reg),

                       use_bias=True)([graph_conv_1, A_in])



mc_1, A_mincut = graph_conv_2, A_in

# doesn't work yet

#diffpool_1, A_mincut, _ = DiffPool(k=64)([graph_conv_22, A_mincut])



graph_conv_21 = GraphConv(64,

                       activation='elu',

                       kernel_regularizer=l2(l2_reg),

                       use_bias=True)([mc_1, A_mincut])



graph_conv_22 = GraphConv(128,

                       activation='elu',

                       kernel_regularizer=l2(l2_reg),

                       use_bias=True)([graph_conv_21, A_mincut])





gap_1 = GlobalAttentionPool(32)(graph_conv_22)

gap_dr = Dropout(0.5)(gap_1)

fc_1 = Dense(16, activation='elu')(gap_dr)

output = Dense(n_out, activation='softmax')(fc_1)



# Build model

model = Model(inputs=[X_in, A_in], outputs=output)

optimizer = Adam(lr=learning_rate)

model.compile(optimizer=optimizer,

              loss='sparse_categorical_crossentropy',

              metrics=['acc'])

model.summary()
from keras.utils.vis_utils import model_to_dot

Image(model_to_dot(model, show_shapes=True).create_png())
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
W, b = model.layers[4].get_weights()

print(W.shape, b.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.imshow(W, vmin=-1, vmax=1, cmap='RdBu')

ax2.bar(np.arange(W.shape[1]), b)
W, b = model.layers[5].get_weights()

print(W.shape, b.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.imshow(W, vmin=-1, vmax=1, cmap='RdBu')

ax2.bar(np.arange(W.shape[1]), b)
Ws = model.layers[6].get_weights()

print([W.shape for W in Ws])

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

ax1.imshow(Ws[0], vmin=-1, vmax=1, cmap='RdBu')

ax2.bar(np.arange(Ws[1].shape[0]), Ws[1])

ax3.imshow(Ws[2], vmin=-1, vmax=1, cmap='RdBu')

ax4.bar(np.arange(Ws[3].shape[0]), Ws[3])
i_model = Model(inputs=[X_in, A_in], outputs=[graph_conv_1, graph_conv_2, graph_conv_21, graph_conv_22])
output_list = i_model.predict(X_test[:32])
fig, m_axs = plt.subplots(4, 5, figsize=(20, 15))

stack_it = lambda x: x.reshape((28, 28, -1)).swapaxes(0, 2).swapaxes(1, 2)

for n_axs, c_data in zip(m_axs.T, [X_test]+output_list):

    for i, c_ax in enumerate(n_axs):

        c_img = stack_it(c_data[i]).squeeze()

        if c_img.ndim==2:

            c_ax.imshow(c_img)

        else:

            c_ax.imshow(montage(c_img), vmin=-0.5, vmax=0.5, cmap='RdBu')

        c_ax.axis('off')