import os

from joblib import Parallel, delayed

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time



from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
FOLDER = '../input/'

OUTPUT = '../input/preprocessed/'

NCORES = 4

os.listdir(FOLDER)
# df_mulliken_charges = pd.read_csv(FOLDER + 'mulliken_charges.csv')

# df_sample =  pd.read_csv(FOLDER + 'sample_submission.csv')

# df_magnetic_shielding_tensors = pd.read_csv(FOLDER + 'magnetic_shielding_tensors.csv')

df_train = pd.read_csv(FOLDER + 'train.csv')

# df_test = pd.read_csv(FOLDER + 'test.csv')

# df_dipole_moments = pd.read_csv(FOLDER + 'dipole_moments.csv')

# df_potential_energy = pd.read_csv(FOLDER + 'potential_energy.csv')

df_structures = pd.read_csv(FOLDER + 'structures.csv')

# df_scalar_coupling_contributions = pd.read_csv(FOLDER + 'scalar_coupling_contributions.csv')
def get_dist_matrix(df_structures, molecule):

    df_temp = df_structures.query('molecule_name == "{}"'.format(molecule))

    locs = df_temp[['x','y','z']].values

    num_atoms = len(locs)

    loc_tile = np.tile(locs.T, (num_atoms,1,1))

    dist_mat = ((loc_tile - loc_tile.T)**2).sum(axis=1)

    return dist_mat
def assign_atoms_index(df, molecule):

    se_0 = df.query('molecule_name == "{}"'.format(molecule))['atom_index_0']

    se_1 = df.query('molecule_name == "{}"'.format(molecule))['atom_index_1']

    assign_idx = pd.concat([se_0, se_1]).unique()

    assign_idx.sort()

    return assign_idx
def get_pickup_dist_matrix(df, df_structures, molecule, num_pickup=5, atoms=['H', 'C', 'N', 'O', 'F']):

    pickup_dist_matrix = np.zeros([0, len(atoms)*num_pickup])

    assigned_idxs = assign_atoms_index(df, molecule) # [0, 1, 2, 3, 4, 5, 6] -> [1, 2, 3, 4, 5, 6]

    dist_mat = get_dist_matrix(df_structures, molecule)

    for idx in assigned_idxs: # [1, 2, 3, 4, 5, 6] -> [2]



        

        dist_arr = dist_mat[idx] # (7, 7) -> (7, )



        atoms_mole = df_structures.query('molecule_name == "{}"'.format(molecule))['atom'].values # ['O', 'C', 'C', 'N', 'H', 'H', 'H']

        atoms_mole_idx = df_structures.query('molecule_name == "{}"'.format(molecule))['atom_index'].values # [0, 1, 2, 3, 4, 5, 6]



        mask_atoms_mole_idx = atoms_mole_idx != idx # [ True,  True, False,  True,  True,  True,  True]

        masked_atoms = atoms_mole[mask_atoms_mole_idx] # ['O', 'C', 'N', 'H', 'H', 'H']

        masked_atoms_idx = atoms_mole_idx[mask_atoms_mole_idx]  # [0, 1, 3, 4, 5, 6]

        masked_dist_arr = dist_arr[mask_atoms_mole_idx]  # [ 5.48387003, 2.15181049, 1.33269675, 10.0578779, 4.34733927, 4.34727838]



        sorting_idx = np.argsort(masked_dist_arr) # [2, 1, 5, 4, 0, 3]

        sorted_atoms_idx = masked_atoms_idx[sorting_idx] # [3, 1, 6, 5, 0, 4]

        sorted_atoms = masked_atoms[sorting_idx] # ['N', 'C', 'H', 'H', 'O', 'H']

        sorted_dist_arr = 1/masked_dist_arr[sorting_idx] #[0.75035825,0.46472494,0.23002898,0.23002576,0.18235297,0.09942455]



        target_matrix = np.zeros([len(atoms), num_pickup])

        for a, atom in enumerate(atoms):

            pickup_atom = sorted_atoms == atom # [False, False,  True,  True, False,  True]

            pickup_dist = sorted_dist_arr[pickup_atom] # [0.23002898, 0.23002576, 0.09942455]

            num_atom = len(pickup_dist)

            if num_atom > num_pickup:

                target_matrix[a, :] = pickup_dist[:num_pickup]

            else:

                target_matrix[a, :num_atom] = pickup_dist

        pickup_dist_matrix = np.vstack([pickup_dist_matrix, target_matrix.reshape(-1)])

    return pickup_dist_matrix
def get_dist_mat(mol):

    assigned_idxs = assign_atoms_index(df_train, mol)

    dist_mat_mole = get_pickup_dist_matrix(df_train, df_structures, mol, num_pickup=num)

    mol_name_arr = [mol] * len(assigned_idxs) 



    return (mol_name_arr, assigned_idxs, dist_mat_mole)
num = 5

mols = df_train['molecule_name'].unique()

dist_mat = np.zeros([0, num*5])

atoms_idx = np.zeros([0], dtype=np.int32)

molecule_names = np.empty([0])



start = time.time()



dist_mats = Parallel(n_jobs=NCORES)(delayed(get_dist_mat)(mol) for mol in mols[:100])

molecule_names = np.hstack([x[0] for x in dist_mats])

atoms_idx = np.hstack([x[1] for x in dist_mats])

dist_mat = np.vstack([x[2] for x in dist_mats])



col_name_list = []

atoms = ['H', 'C', 'N', 'O', 'F']

for a in atoms:

    for n in range(num):

        col_name_list.append('dist_{}_{}'.format(a, n))

        

se_mole = pd.Series(molecule_names, name='molecule_name')

se_atom_idx = pd.Series(atoms_idx, name='atom_index')

df_dist = pd.DataFrame(dist_mat, columns=col_name_list)

df_distance = pd.concat([se_mole, se_atom_idx,df_dist], axis=1)



elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
# df_distance.to_csv(OUTPUT + 'distance1000.csv', index=False)
# df_dist = pd.read_csv(OUTPUT + 'distance1000.csv')

df_distance.head()
def merge_atom(df, df_distance):

    df_merge_0 = pd.merge(df, df_distance, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])

    df_merge_0_1 = pd.merge(df_merge_0, df_distance, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

    del df_merge_0_1['atom_index_x'], df_merge_0_1['atom_index_y']

    return df_merge_0_1
start = time.time()

df_train_dist = merge_atom(df_train, df_distance) # corrected!: df_dist -> df_distance

elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
# df_train_dist.to_csv(OUTPUT + 'train_dist1000.csv', index=False)
# df_train_dist = pd.read_csv(OUTPUT + 'train_dist1000.csv')

df_train_dist.head()
df_1JHC = df_train_dist.query('type == "1JHC"')

y = df_1JHC['scalar_coupling_constant'].values

X = df_1JHC[df_1JHC.columns[5:]].values

print(X.shape)

print(y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
mlp = MLPRegressor(hidden_layer_sizes=(100,50))

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_val)

plt.scatter(y_val, y_pred)

plt.title('1JHC')

plt.plot([80, 200], [80, 200])

plt.show()