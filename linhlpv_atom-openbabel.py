# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
!conda install -c openbabel openbabel -y
import openbabel

import os

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

from tqdm import *

path = '../input/champsscalarold/structures/'



obConversion = openbabel.OBConversion()

obConversion.SetInFormat('xyz')



def convert_all_xyzfile_to_mol():

    obConversion = openbabel.OBConversion()

    obConversion.SetInFormat("xyz")



    structures_path = '../input/champsscalarold/structures/'

    mols = {}

    mol_files = sorted(os.listdir(structures_path))

    mols_index = dict(map(reversed, enumerate(mol_files)))

    with tqdm(total=len(mol_files)) as pbar:



        for i, name in enumerate(mol_files):

            mol = openbabel.OBMol()

            obConversion.ReadFile(mol, structures_path+name)

            mols[name[:len(name) - 4]] = mol

            pbar.update()

            # if i > 4:

            #     break

    # print(mols)

    return mols, mols_index
mols, mols_index = convert_all_xyzfile_to_mol()
import networkx as nx



def mol_to_nx(mol):

    G = nx.Graph()

    n_atoms = mol.NumAtoms()

    n_bonds = mol.NumBonds()

#     print(n_atoms, n_bonds)



    for atom_idx in range(n_atoms):

        atom = mol.GetAtomById(atom_idx)

        G.add_node(atom.GetIdx(), 

                   atomic_num= atom.GetAtomicNum(),

                   formal_charge=atom.GetFormalCharge(),

                   is_chiral=atom.IsChiral(),

                   hybridization=atom.GetHyb(),

                   is_aromatic=atom.IsAromatic(),

                   position=[atom.GetX(), atom.GetY(), atom.GetZ()],

                   is_ring=atom.IsInRing(),

                   is_ring_3=atom.IsInRingSize(3),

                   is_ring_4=atom.IsInRingSize(4),

                   is_ring_5=atom.IsInRingSize(5),

                   is_ring_6=atom.IsInRingSize(6),

                   is_ring_7=atom.IsInRingSize(7),

                   is_ring_8=atom.IsInRingSize(8)

                   )



    for bond_idx in range(n_bonds):

        bond = mol.GetBond(bond_idx)

        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondOrder(), is_aromatic=bond.IsAromatic(), in_ring=bond.IsInRing(), bond_length=bond.GetLength())

    

    return G
import pickle

graph_mol = {}

node_features = []

for name in mols.keys():

    mol = mols[name]

    G = mol_to_nx(mol)

    graph_mol[name] = G

    for i in G.nodes:

        node = G.nodes[i]

        n = [name, i-1, node['atomic_num'], node['formal_charge'], node['is_chiral'], node['hybridization'], node['is_aromatic'], node['position'][0], node['position'][1], node['position'][2],

               node['is_ring'], node['is_ring_3'], node['is_ring_4'], node['is_ring_5'], node['is_ring_6'], node['is_ring_7'], node['is_ring_8']]

        node_features.append(n)



node_features = np.array(node_features)

print(node_features.shape)

with open('graph_mol_openbabel_v1.pkl', 'wb')  as f:

    pickle.dump(graph_mol, f)



print('Done')
node_df = pd.DataFrame(node_features, columns=['molecule_name', 'atom_index', 'atomic_num', 'formal_charge', 'is_chiral', 'hybridization', 'is_aromatic', 'x', 'y', 'z', 'is_ring', 'is_ring_3', 'is_ring_4', 'is_ring_5', 'is_ring_6', 'is_ring_7', 'is_ring_8'])
node_df
node_df.to_csv('node_features.csv', index=False)
mol = mols["dsgdb9nsd_000001"]
mol
C = mol.GetAtomById(0)

H = mol.GetAtomById(1)
C.GetAtomicNum()
b1 = C.GetBond(H)

b2 = H.GetBond(C)
b1.GetLength()
b2.GetLength()
b1 == b2
b1.GetBeginAtom().GetAtomicNum()
b2.GetBeginAtom().GetAtomicNum()
b2.GetBondOrder()
b1.GetEquibLength()
b2.GetEquibLength()
nbr = []

G = graph_mol['dsgdb9nsd_133882']
for name in graph_mol.keys():

    G = graph_mol[name]

    for i in G.nodes:

        nbr_idx = [a for a in G.neighbors(i)]

        nbr_atomic_num = [G.nodes[idx]['atomic_num'] for idx in nbr_idx]

        n_H = sum([_ == 1 for _ in nbr_atomic_num])

        n_C = sum([_ == 6 for _ in nbr_atomic_num])

        n_O = sum([_ == 8 for _ in nbr_atomic_num])

        n_F = sum([_ == 9 for _ in nbr_atomic_num])

        n_N = sum([_ == 7 for _ in nbr_atomic_num])



        nbr_info = [name, i-1, n_H, n_C, n_O, n_F, n_N]

        nbr.append(nbr_info)
nbr_df = pd.DataFrame(nbr, columns=['molecule_name', 'atom_index', 'n_H', 'n_C', 'n_O', 'n_F', 'n_N'])
nbr_df.to_csv('neighbors.csv', index=False)
nbr_df