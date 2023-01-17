import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from ase import Atoms

import ase.visualize  # clickable 3D molecule viewer    # pip install ase



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import os

!ls -al --color ../input
def load_dir_csv(directory):

    csv_files = sorted( [ f for f in os.listdir(directory) if f.endswith(".csv") ])    

    csv_vars  = [ filename[:-4] for filename in csv_files ]

    gdict = globals()

    for filename, var in zip( csv_files, csv_vars ):

        print(f"{var:32s} = pd.read_csv({directory}/{filename})")

        gdict[var] = pd.read_csv( f"{directory}/{filename}" )

        print(f"{'nb of rows ':32s} = " + str(len(gdict[var])))

        display(gdict[var].head())



load_dir_csv("../input/champs-scalar-coupling")

load_dir_csv("../input/predicting-molecular-properties-bonds/")

                       
structures.atom.unique()
train.type.unique()
def view3d_molecule(name, xsize="200px", ysize="200px"):

    """Mouse clickeble 3D view"""

    m = structures[structures.molecule_name == name]

    positions = m[['x','y','z']].values

    v = ase.visualize.view(Atoms(positions=positions, symbols=m.atom.values), 

                           viewer="x3d") 

    return v



cpk = { 

    'C': ("black", 2),

    'H': ("white", 1),

    'O': ("red",   2),

    'N': ("dodgerblue", 2),

    'F': ("green", 2) }



bond_colors = {'1.0':'black', '1.5':'darkgreen', '2.0':'green', '3.0':'red'}



def bond_type_to_pair(bond_type):

    return bond_type[3:]

def bond_type_to_n(bond_type):

    return bond_type[0:3]



def plot_molecule(name, ax=None, bonds=None, charges=None, elev=0, azim=-60):

    """bonds = if provided add bonds display from the bond table dataset in https://www.kaggle.com/asauve/predicting-molecular-properties-bonds

    elev = 3D elevation angle [degree] for the molecule view

    azim = 3D azimut angle [degree]

    """

    if not ax:

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

    if (elev != 0) or (azim != -60):

        ax.view_init(elev=elev, azim=azim)

    

    # atoms location

    m = structures[structures.molecule_name == name].sort_values(by='atom_index')

    if (charges is not None):

        charges = charges[charges.molecule_name == name].sort_values(by='atom_index')

        if len(charges) != len(m):

            print(f"Warning bad charges data for molecule {name}")

    

    # formula

    acount = {a : 0 for a in cpk}

    for a in m.atom:

        acount[a] += 1

    formula = ""

    for a in acount:

        if acount[a] == 1:

            formula += a

        elif acount[a] > 1:

            formula += "%s_{%d}" % (a, acount[a])



    ax.set_title(f'{name} ${formula}$')

    

    # display couplings (coupling is not molecular bonds!)

    couples = train[train.molecule_name == name][['atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]

    for c in couples.itertuples():

        m0 = m[m.atom_index == c.atom_index_0]

        m1 = m[m.atom_index == c.atom_index_1]

        ax.plot([float(m0.x), float(m1.x)],[float(m0.y), float(m1.y)],[float(m0.z), float(m1.z)],

               linestyle = ['', '-', '--', 'dotted'][int(c.type[0])],

               color     = ['', 'black', 'green', 'red' ][int(c.type[0])],

               linewidth = abs(float(c.scalar_coupling_constant))/5,

               alpha=0.2), 

    if bonds is not None:

        for b in bonds[bonds.molecule_name == name].itertuples():

            m0 = m[m.atom_index == b.atom_index_0]

            m1 = m[m.atom_index == b.atom_index_1]

            ax.plot([float(m0.x), float(m1.x)],[float(m0.y), float(m1.y)],[float(m0.z), float(m1.z)], 'black', 

                    linewidth=2*float(b.nbond),

                    color=bond_colors[bond_type_to_n(b.bond_type)])

            

    # display atoms

    ax.scatter(m.x, m.y, m.z, c=[cpk[a][0] for a in m.atom], s=[100*cpk[a][1] for a in m.atom], edgecolor='black')

        

    # display atom index and charges

    for row in m.itertuples():

        x = float(row.x) - 0.15 if row.x > ax.get_xlim()[0] + 0.15 else row.x

        y = float(row.y) - 0.15 if row.y > ax.get_ylim()[0] + 0.15 else row.y

        z = float(row.z) - 0.15 if row.z > ax.get_zlim()[0] + 0.15 else row.z

        ax.text(x, y, z, str(row.atom_index), color='darkviolet')

        if charges is not None:

            ch = float(charges[charges.atom_index == row.atom_index].charge)

            if ch != 0:

                x = float(row.x) + 0.15 if row.x < ax.get_xlim()[1] - 0.15 else row.x

                y = float(row.y) + 0.15 if row.y > ax.get_ylim()[1] - 0.15 else row.y

                z = float(row.z) + 0.15 if row.z < ax.get_zlim()[1] - 0.15 else row.z

                ax.text(x, y, z, f"{ch:+.1f}", color='orangered' if ch > 0 else 'blue',

                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, 

                                  edgecolor='black'))

                

ax = plot_molecule("dsgdb9nsd_000007", bonds=train_bonds)
view3d_molecule("dsgdb9nsd_000007")
nrow = 5

ncol = 4

fig = plt.figure(figsize=(20, 20))

molecules = train.molecule_name.unique()

for i in range(nrow*ncol):

    ax = fig.add_subplot(nrow, ncol, 1+i, projection='3d')

    plot_molecule(molecules[i], ax=ax, bonds=train_bonds)
view3d_molecule("dsgdb9nsd_000023")
ionized = train_charges[(train_charges.charge != 0)].molecule_name.unique()



# filter out molecules with failed bonding

errors  = train_bonds[train_bonds.error == 1].molecule_name.unique()

errors  = {e:1 for e in errors} # convert to dict for fast lookup

ionized = [ name for name in ionized if not name in errors]



nrow = 3

ncol = 3

fig = plt.figure(figsize=(18, 18))

for i in range(nrow*ncol):

    ax = fig.add_subplot(nrow, ncol, 1+i, projection='3d')

    plot_molecule(ionized[i], ax=ax, bonds=train_bonds, charges=train_charges)
view3d_molecule("dsgdb9nsd_076394")