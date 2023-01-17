import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



import os

print(os.listdir("../input"))



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
structures = pd.read_csv("../input/structures.csv")

structures.head()
M = 8000

fig, ax = plt.subplots(1,3,figsize=(20,5))



colors = ["darkred", "dodgerblue", "mediumseagreen", "gold", "purple"]

atoms = structures.atom.unique()



for n in range(len(atoms)):



    ax[0].scatter(structures.loc[structures.atom==atoms[n]].x.values[0:M],

                  structures.loc[structures.atom==atoms[n]].y.values[0:M],

                  color=colors[n], s=2, alpha=0.5, label=atoms[n])

    ax[0].legend()

    ax[0].set_xlabel("x")

    ax[0].set_ylabel("y")

    

    ax[1].scatter(structures.loc[structures.atom==atoms[n]].x.values[0:M],

                  structures.loc[structures.atom==atoms[n]].z.values[0:M],

                  color=colors[n], s=2, alpha=0.5, label=atoms[n])

    ax[1].legend()

    ax[1].set_xlabel("x")

    ax[1].set_ylabel("z")

    

    ax[2].scatter(structures.loc[structures.atom==atoms[n]].y.values[0:M],

                  structures.loc[structures.atom==atoms[n]].z.values[0:M],

                  color=colors[n], s=2, alpha=0.5, label=atoms[n])

    ax[2].legend()

    ax[2].set_xlabel("y")

    ax[2].set_ylabel("z")
M = 200000

fig, ax = plt.subplots(1,3,figsize=(20,5))



ax[0].scatter(structures.x.values[0:M],

              structures.y.values[0:M],

              c = structures.index.values[0:M],

              s=2, alpha=0.5, cmap="magma")

ax[0].set_xlabel("x")

ax[0].set_xlabel("y");



ax[1].scatter(structures.x.values[0:M],

              structures.z.values[0:M],

              c = structures.index.values[0:M],

              s=2, alpha=0.5, cmap="magma")

ax[1].set_xlabel("x")

ax[1].set_xlabel("z");



ax[2].scatter(structures.y.values[0:M],

              structures.z.values[0:M],

              c = structures.index.values[0:M],

              s=2, alpha=0.5, cmap="magma")

ax[2].set_xlabel("y")

ax[2].set_xlabel("z");
N = 100000



trace1 = go.Scatter3d(

    x=structures.x.values[0:N], 

    y=structures.y.values[0:N],

    z=structures.z.values[0:N],

    mode='markers',

    marker=dict(

        color=structures.index.values[0:N],

        colorscale = "YlGnBu",

        opacity=0.3,

        size=1

    )

)



figure_data = [trace1]

layout = go.Layout(

    title = 'Coloring by index',

    scene = dict(

        xaxis = dict(title="x"),

        yaxis = dict(title="y"),

        zaxis = dict(title="z"),

    ),

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ),

    showlegend=True

)



fig = go.Figure(data=figure_data, layout=layout)

py.iplot(fig, filename='simple-3d-scatter')
molecules_size = structures.groupby("molecule_name").atom.size()

mean_size = molecules_size.rolling(window=500).mean()
plt.figure(figsize=(20,5))

plt.plot(molecules_size.values, '+')

plt.plot(np.arange(250, len(molecules_size)-249), mean_size.dropna().values, '-')

plt.xlabel("Molecule name number")

plt.ylabel("Number of atoms");
structures["atom_count"] = structures.groupby("molecule_name").atom.transform("size")
to_show = ["atom_index", "atom_count"]

my_choice = to_show[0]
M = 10000

fig, ax = plt.subplots(5,3, figsize=(20,25))



for n in range(len(atoms)):

    ax[n,0].scatter(structures.loc[structures.atom==atoms[n]].x.values[0:M],

                    structures.loc[structures.atom==atoms[n]].y.values[0:M],

                    c=structures.loc[structures.atom==atoms[n]][my_choice].values[0:M],

                    s=5, alpha=0.5, cmap="inferno")

    ax[n,0].set_title(atoms[n])

    ax[n,0].set_xlabel("x")

    ax[n,0].set_ylabel("y")

    

    ax[n,1].scatter(structures.loc[structures.atom==atoms[n]].x.values[0:M],

                    structures.loc[structures.atom==atoms[n]].z.values[0:M],

                    c=structures.loc[structures.atom==atoms[n]][my_choice].values[0:M],

                    s=5, alpha=0.5, cmap="inferno")

    ax[n,1].set_title(atoms[n])

    ax[n,1].set_xlabel("x")

    ax[n,1].set_ylabel("z")

    

    ax[n,2].scatter(structures.loc[structures.atom==atoms[n]].y.values[0:M],

                    structures.loc[structures.atom==atoms[n]].z.values[0:M],

                    c=structures.loc[structures.atom==atoms[n]][my_choice].values[0:M],

                    s=5, alpha=0.5, cmap="inferno")

    ax[n,2].set_title(atoms[n])

    ax[n,2].set_xlabel("y")

    ax[n,2].set_ylabel("z")
mulliken = pd.read_csv("../input/mulliken_charges.csv")
train_structures = structures.loc[structures.molecule_name.isin(train.molecule_name)].copy()

train_structures = train_structures.set_index(["molecule_name", "atom_index"])

mulliken = mulliken.set_index(["molecule_name", "atom_index"])

mulliken["atom"] = train_structures.atom



plt.figure(figsize=(20,5))

sns.violinplot(x=mulliken.atom, y=mulliken.mulliken_charge)

plt.title("How are the mulliken charges distributed given the atom type?");
train_structures["mulliken"] = mulliken.mulliken_charge

train_structures.head()
M = 100000

fig, ax = plt.subplots(5,1,figsize=(20,100))



for n in range(len(atoms)):

    try:

        ax[n].scatter(train_structures.loc[train_structures.atom==atoms[n]].x.values[0:M],

                      train_structures.loc[train_structures.atom==atoms[n]].z.values[0:M], s=1,alpha=0.5,

                      c=train_structures.loc[train_structures.atom==atoms[n]].mulliken.values[0:M],

                      cmap="coolwarm")

    except:

        ax[n].scatter(train_structures.loc[train_structures.atom==atoms[n]].x.values,

                      train_structures.loc[train_structures.atom==atoms[n]].z.values, s=1, alpha=0.5,

                      c=train_structures.loc[train_structures.atom==atoms[n]].mulliken.values,

                      cmap="coolwarm")

    ax[n].set_title("Mulliken charges of " + str(atoms[n]))

    ax[n].set_xlabel("x")

    ax[n].set_ylabel("z")
dipoles = pd.read_csv("../input/dipole_moments.csv")

dipoles.head()
test.molecule_name.isin(dipoles.molecule_name).sum()
plt.figure(figsize=(20,5))

sns.distplot(dipoles.X, label="X")

sns.distplot(dipoles.Y, label="Y")

sns.distplot(dipoles.Z, label="Z")

plt.legend();

plt.title("Coordinate specific dipole moment distribution");
dipoles["total"] = np.abs(dipoles.X) + np.abs(dipoles.Y) + np.abs(dipoles.Z)
plt.figure(figsize=(20,5))

sns.distplot(dipoles.total);

plt.title("How is the absolute sum of X, Y, Z dipole moments distributed?");
fig, ax = plt.subplots(3,1,figsize=(20,60))

ax[0].scatter(dipoles.X.values, dipoles.Y.values, s=0.5, alpha=0.1);

ax[0].set_xlabel("X")

ax[0].set_ylabel("Y")

ax[1].scatter(dipoles.X.values, dipoles.Z.values, s=0.5, alpha=0.1);

ax[1].set_xlabel("X")

ax[1].set_ylabel("Z")

ax[2].scatter(dipoles.Y.values, dipoles.Z.values, s=0.5, alpha=0.1);

ax[2].set_xlabel("Y")

ax[2].set_ylabel("Z");
energy = pd.read_csv("../input/potential_energy.csv")

energy.head()
atom_counts = structures[structures.molecule_name.isin(dipoles.molecule_name)].groupby("molecule_name").atom.size()
max_mulliken = mulliken.groupby("molecule_name").mulliken_charge.max()

min_mulliken = mulliken.groupby("molecule_name").mulliken_charge.min()
fig, ax = plt.subplots(1,3,figsize=(20,5))

ax[0].scatter(energy.potential_energy.values, dipoles.total.values, s=1, c=atom_counts.values, cmap="YlGnBu");

ax[0].set_xlabel("Potential energy")

ax[0].set_ylabel("Total absolute dipole moment")

ax[0].set_title("Colored by atom counts of a molecule");



ax[1].scatter(energy.potential_energy.values, max_mulliken.values, c=atom_counts.values, cmap="YlGnBu", s=1);

ax[1].set_xlabel("Potential energy")

ax[1].set_ylabel("Max mulliken charge");



ax[2].scatter(energy.potential_energy.values, min_mulliken.values, c=atom_counts.values, cmap="YlGnBu", s=1);

ax[2].set_xlabel("Potential energy")

ax[2].set_ylabel("Min mulliken charge");