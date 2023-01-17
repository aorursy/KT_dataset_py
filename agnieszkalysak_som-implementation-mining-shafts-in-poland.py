import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import StandardScaler

!pip install MiniSom

!pip install --upgrade pip

from minisom import MiniSom
dataset = pd.read_csv("../input/mines-data/mines_data.csv", encoding='latin1')

dataset
data_som = dataset[['section_area', 'object_area', 'rel_height', 'object_distance']]

data_som
sc = StandardScaler()

data_som_sc = sc.fit_transform(data_som)
ids = dataset[['ID']]

I = ids['ID'].tolist()

section = dataset[['section']]

S = section['section'].tolist()
som = MiniSom(x = 21, y = 21, input_len = 4, neighborhood_function = 'gaussian', sigma = 1.5, learning_rate = 0.5)

som.random_weights_init(data_som_sc)

som.train_random(data = data_som_sc, num_iteration = 1000)
from pylab import bone, pcolor, colorbar, plot, show

bone()

plt.figure(figsize=(14, 10))

pcolor(som.distance_map().T, alpha=.9)

colorbar()

markers = ['*', 'o', 's', 'D', 'X', 'v', 'P', 'h', '^']

colors = ['grey', 'r', 'g', 'b', 'y', 'c', 'orange', 'fuchsia', 'lime']

for i, x in enumerate(data_som_sc):

    w = som.winner(x)

    plot(w[0] + 0.5,

        w[1] + 0.5,

        markers[S[i]],

        markeredgecolor=colors[S[i]],

        markerfacecolor='None',

        markersize = 10,

        markeredgewidth=2

        )

show()
from matplotlib.lines import Line2D



legend_elements = [Line2D([0], [0], marker='o', color='r', label='1',

                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),

                   Line2D([0], [0], marker='s', color='g', label='2',

                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),

                   Line2D([0], [0], marker='D', color='b', label='3',

                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),

                   Line2D([0], [0], marker='X', color='y', label='4',

                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),

                   Line2D([0], [0], marker='v', color='c', label='5',

                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),

                   Line2D([0], [0], marker='P', color='orange', label='6',

                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),

                   Line2D([0], [0], marker='h', color='fuchsia', label='7',

                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2),

                   Line2D([0], [0], marker='^', color='lime', label='8',

                   markerfacecolor='w', markersize=10, linestyle='None', markeredgewidth=2)]



from pylab import bone, pcolor, colorbar, plot, show

bone()

plt.figure(figsize=(14, 10))

pcolor(som.distance_map().T, cmap='gray_r', alpha=.7)

colorbar()

markers = ['*', 'o', 's', 'D', 'X', 'v', 'P', 'h', '^']

colors = ['grey', 'r', 'g', 'b', 'y', 'c', 'orange', 'fuchsia', 'lime']

for i, x in enumerate(data_som_sc):

    w = som.winner(x)

    plot(w[0] + 0.5,

        w[1] + 0.5,

        markers[S[i]],

        markeredgecolor=colors[S[i]],

        markerfacecolor='None',

        markersize = 10,

        markeredgewidth=2

        )

plt.legend(handles=legend_elements, bbox_to_anchor=(1.27, 1.01), prop={'size': 12}, ncol=1)

show()
from pylab import bone, pcolor, colorbar, plot, show

bone()

plt.figure(figsize=(14, 10))

pcolor(som.distance_map().T, cmap='gray_r', alpha=.7)

colorbar()

markers = ['*', 'o', 's', 'D', 'X', 'v', 'P', 'h', '^']

colors = ['grey', 'r', 'g', 'b', 'y', 'c', 'orange', 'fuchsia', 'lime']

for i, x in enumerate(data_som_sc):

    w = som.winner(x)

    plot(w[0] + 0.5,

        w[1] + 0.5,

        markers[S[i]],

        markeredgecolor=colors[S[i]],

        markerfacecolor='None',

        markersize = 10,

        markeredgewidth=2

        )

wmap = {}

im = 0

for x, t in zip(data_som_sc, I):

    w = som.winner(x)

    wmap[w] = im

    plt. text(w[0]+.8,  w[1]+.3,  str(t),

              color='k', fontdict={'weight': 'normal', 'size': 14})

    im = im + 1

plt.legend(handles=legend_elements, bbox_to_anchor=(1.27, 1.01), prop={'size': 12}, ncol=1)

show()
ro1 = dataset[dataset['ID'] == 0]

ro1
x = dataset.object_area

plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

sns.distplot(x, label="item_id", kde=True, bins=20)
x = dataset.rel_height

plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

sns.distplot(x, label="item_id", kde=True, bins=20)
x = dataset.object_distance

plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

sns.distplot(x, label="item_id", kde=True, bins=20)
dgs2 = dataset[dataset['section'] == 2]

dgs2
x = dataset.rel_height

plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

sns.distplot(x, label="x", kde=False, bins=20)
bd3 = dataset[dataset['section'] == 3]

bd3
x = irregular.object_distance

plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

sns.distplot(x, label="x", kde=False, bins=5)
bd3_upper_right = bd3[bd3['object_distance'] > 30]

bd3_upper_right
bd3_right = bd3[bd3['object_distance'] > 20]

bd3_right
bd3_lower = bd3[bd3['object_distance'] > 15]

bd3_lower
bd3_left = bd3[bd3['object_distance'] < 10]

bd3_left
yx4 = dataset[dataset['section'] == 4]

yx4
yx4_right = yx4[yx4['object_distance'] > 10]

yx4_right
yx4_lower = yx4_upper[yx4_upper['object_distance'] > 5]

yx4_lower
yx4_upper = yx4[yx4['object_distance'] < 10]

yx4_upper
ct5 = dataset[dataset['section'] == 5]

ct5
oc6 = dataset[dataset['section'] == 6]

oc6
fh7 = dataset[dataset['section'] == 7]

fh7
bgt8 = dataset[dataset['section'] == 8]

bgt8