import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.spatial import distance_matrix



from sklearn import preprocessing

import os

print(os.listdir("../input"))



datadir = "../input/"
train = pd.read_csv(datadir + 'champs-scalar-coupling/train.csv')

test = pd.read_csv(datadir + 'champs-scalar-coupling/test.csv')

structures = pd.read_csv(datadir + 'champs-scalar-coupling/structures.csv')
train_bonds = pd.read_csv(datadir + 'predicting-molecular-properties-bonds/train_bonds.csv')

test_bonds = pd.read_csv(datadir + 'predicting-molecular-properties-bonds/test_bonds.csv')
angs = pd.read_csv(datadir + "angle-and-dihedral-for-the-champs-structures/angles.csv")
scale_min  = train['scalar_coupling_constant'].min()

scale_max  = train['scalar_coupling_constant'].max()

scale_mid = (scale_max + scale_min)/2

scale_norm = scale_max - scale_mid



train['scalar_coupling_constant'] = (train['scalar_coupling_constant'] - scale_mid)/scale_norm



# One hot encoding gets  too big for Kaggle, let's try label

# use npz now, back to OH

train[['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']] =  pd.get_dummies(train['type'])

test[['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']]  =  pd.get_dummies(test['type'])



#le = preprocessing.LabelEncoder()

#le.fit(['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN'])

#train['l_type'] = (le.transform(train['type']) + 1)/8.

#test['l_type'] = (le.transform(test['type']) + 1)/8.
structures[['C', 'F' ,'H', 'N', 'O']] = pd.get_dummies(structures['atom'])

structures[['x', 'y', 'z']] = structures[['x', 'y', 'z']]/10.
test_bonds[['nbond_1', 'nbond_1.5', 'nbond_2', 'nbond_3']] = pd.get_dummies(test_bonds['nbond'])#test_bonds['nbond']/3

train_bonds[['nbond_1', 'nbond_1.5', 'nbond_2', 'nbond_3']] = pd.get_dummies(train_bonds['nbond'])#train_bonds['nbond']/3

angs['dihedral'] = angs['dihedral']/np.pi

# Should I rather one-hot this?

angs['shortest_path_n_bonds'] = angs['shortest_path_n_bonds']/6.0

angs = angs.fillna(0)
train_mol_names = train['molecule_name'].unique()

test_mol_names  = test['molecule_name'].unique()



train_structures = structures.loc[structures['molecule_name'].isin(train_mol_names)]

test_structures = structures.loc[structures['molecule_name'].isin(test_mol_names)]



train_struct_group = train_structures.groupby('molecule_name')

test_struct_group  = test_structures.groupby('molecule_name')



train_group = train.groupby('molecule_name')

test_group  = test.groupby('molecule_name')



train_bond_group = train_bonds.groupby('molecule_name')

test_bond_group  = test_bonds.groupby('molecule_name')



train_angs = angs.loc[angs['molecule_name'].isin(train_mol_names)]

test_angs = angs.loc[angs['molecule_name'].isin(test_mol_names)]



train_angs_group = train_angs.groupby('molecule_name')

test_angs_group  = test_angs.groupby('molecule_name')



# Find max nodes in graph:

max_size = train_struct_group.size().max()
# Values our nodes will have

node_vals = ['C', 'F' ,'H', 'N', 'O']#, 'x', 'y', 'z']

#Values our edges will have (minus distance, for now)

bond_vals = ['nbond_1', 'nbond_1.5', 'nbond_2', 'nbond_3']#['nbond']

j_coup_vals = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']#'l_type']

ang_vals = ['shortest_path_n_bonds','cosinus','dihedral']

edge_vals = j_coup_vals + bond_vals + ang_vals



# Find amount of training molecules

n_train_mols = len(train_mol_names)

n_test_mols = len(test_mol_names)



# Find dim of edges and nodes

bond_dim  = len(bond_vals)

j_coup_dim= len(j_coup_vals)

ang_dim   = len(ang_vals)

node_dim  = len(node_vals)

edge_dim  = len(edge_vals) 



# Additional edge dims for distances 

add_edge_dim = 1
train_nodes_array     = np.zeros((n_train_mols, max_size, node_dim), dtype=np.float32) 

train_in_edges_array  = np.zeros((n_train_mols, max_size, max_size, edge_dim + add_edge_dim),dtype=np.float32) 

train_out_edges_array = np.zeros((n_train_mols, max_size, max_size, 1),dtype=np.float32) 



test_nodes_array     = np.zeros((n_test_mols, max_size, node_dim), dtype=np.float32) 

test_in_edges_array  = np.zeros((n_test_mols, max_size, max_size, edge_dim + add_edge_dim),dtype=np.float32) 
def make_arrs(val_group, struct_group, bond_group, ang_group, test):

    i = 0

    for values, structs, bonds, angles in zip(val_group, struct_group, bond_group, ang_group):

        if (not i%1000):

            print(i)



        # Calculate distances

        distances = np.zeros((max_size, max_size, 1))

        coords = structs[1][['x','y','z']].values

        dists  = distance_matrix(coords, coords)

        distances[:dists.shape[0],:dists.shape[1], 0] = dists 

        

        # Create nodes

        mol_info = structs[1][node_vals].values

        nodes = np.zeros((max_size, node_dim))

        nodes[:mol_info.shape[0], :mol_info.shape[1]] = mol_info



        # Create edges

        in_feats = np.zeros((max_size, max_size, j_coup_dim))

        ind = values[1][['atom_index_0', 'atom_index_1' ]].values

        in_feats[ind[:,0], ind[:,1], 0:j_coup_dim] = values[1][j_coup_vals].values

        in_feats[ind[:,1], ind[:,0], 0:j_coup_dim] = in_feats[ind[:,0], ind[:,1], 0:j_coup_dim]



        # Create bonds

        in_bonds = np.zeros((max_size, max_size, bond_dim))

        ind_bonds = bonds[1][['atom_index_0', 'atom_index_1' ]].values

        in_bonds[ind_bonds[:,0], ind_bonds[:,1]] = bonds[1][bond_vals].values

        in_bonds[ind_bonds[:,1], ind_bonds[:,0]] = in_bonds[ind_bonds[:,0], ind_bonds[:,1]]

        

        # Create angles

        ind_angs = angles[1][['atom_index_0', 'atom_index_1' ]].values

        ang_mat  = np.zeros((max_size, max_size, ang_dim))

        ang_mat[ind_angs[:,0], ind_angs[:,1]]  = angles[1][ang_vals]

        ang_mat[ind_angs[:,1], ind_angs[:,0]]  = ang_mat[ind_angs[:,0], ind_angs[:,1]]

        

        # concat all edge values 

        in_edges = np.concatenate((in_feats, in_bonds, ang_mat, distances),axis=2)







        

        if not test:           

            out_edges = np.zeros((max_size, max_size, 1))

            out_edges[ind[:,0], ind[:,1], 0] = values[1]['scalar_coupling_constant' ].values

            out_edges[ind[:,1], ind[:,0], 0] = out_edges[ind[:,0], ind[:,1], 0]

        



            train_nodes_array[i]      = nodes

            train_in_edges_array[i]   = in_edges

            train_out_edges_array[i]  = out_edges

        else:

            test_nodes_array[i]      = nodes

            test_in_edges_array[i]   = in_edges

        i = i + 1

make_arrs(train_group, train_struct_group, train_bond_group, train_angs_group, test = False)
make_arrs(test_group, test_struct_group, test_bond_group, test_angs_group, test = True)
np.savez_compressed("nodes_train.npz" , train_nodes_array)

np.savez_compressed("in_edges_train.npz" , train_in_edges_array)

np.savez_compressed("out_edges_train.npz" , train_out_edges_array)



np.savez_compressed("nodes_test.npz" , test_nodes_array)

np.savez_compressed("in_edges_test.npz" , test_in_edges_array)