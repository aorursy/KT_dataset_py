import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



plt.rcParams["patch.force_edgecolor"] = True

#rc('text', usetex=True)

from IPython.display import display # Allows the use of display() for DataFrames

import seaborn as sns

sns.set(style="whitegrid", palette="muted", color_codes=True)

sns.set_style("whitegrid", {'grid.linestyle': '--'})

red = sns.xkcd_rgb["light red"]

green = sns.xkcd_rgb["medium green"]

blue = sns.xkcd_rgb["denim blue"]
# Data Loading

structures = pd.read_csv('../input/structures.csv')

train = pd.read_csv('../input/train.csv')

train["target_exp"] = np.log1p(np.exp(train.scalar_coupling_constant))/10 # for line width
train.head()
def draw_3d_graph(struct, train_df, ax):

    ax.scatter(xs=struct['x'], ys=struct['y'], zs=struct['z'], s=100, color="r")

    ax.set_title(f'{train_df.iloc[0].molecule_name}, n:{train_df.shape[0]}')

    

    for i, tr in train_df.iterrows():

        g0 = struct.iloc[tr.atom_index_0]

        g1 = struct.iloc[tr.atom_index_1]

        ax.plot3D([g0['x'], g1['x']], [g0['y'], g1['y']], [g0['z'], g1['z']], lw=tr.target_exp, color="gray", alpha=0.5)





n_row = 50

n_col = 4

cnt = 0

plt.figure(figsize=(25, 5*n_row))

for i, g in structures.groupby("molecule_name"):

    data = train[train.molecule_name==i]

    if data.shape[0]==0: continue

    ax = plt.subplot(n_row, n_col, cnt+1, projection='3d')

    draw_3d_graph(g, data, ax)

    cnt += 1

    if cnt == n_row*n_col:

        break

plt.tight_layout()

plt.show()
mol_size = structures.groupby("molecule_name").size() 

mol_size_large = mol_size[mol_size>15]



n_row = 50

n_col = 4

cnt = 0

plt.figure(figsize=(25, 5*n_row))

for i, g in structures[structures.molecule_name.isin(mol_size_large.index.tolist())].groupby("molecule_name"):

    data = train[train.molecule_name==i]

    if data.shape[0]==0: continue

    ax = plt.subplot(n_row, n_col, cnt+1, projection='3d')

    draw_3d_graph(g, data, ax)

    cnt += 1

    if cnt == n_row*n_col:

        break

plt.tight_layout()

plt.show()