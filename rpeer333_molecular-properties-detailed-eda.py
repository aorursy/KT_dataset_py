%%capture



import os

from os.path import join



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.spatial.distance import euclidean

from scipy import stats

from tqdm import tqdm_notebook

import time





try:

    import ase

except:

    !pip install ase

    import ase

from ase import Atoms

import ase.visualize
DATA_DIR = '../input'



structure_dir = join(DATA_DIR, 'structures')

os.listdir(DATA_DIR)
train = pd.read_csv(join(DATA_DIR, 'train.csv'))

test = pd.read_csv(join(DATA_DIR, 'test.csv'))

structures = pd.read_csv(join(DATA_DIR, 'structures.csv'))



print(f'number of structures: {len(os.listdir(structure_dir))}')

print(f'training interactions: {len(train)}')

print(f'test interactions: {len(test)}')

print(f'fraction of interactions used for testing: {round(len(test) / (len(test) + len(train)), 2)}')

train.head()
assert len(structures.molecule_name.unique()) == len(os.listdir(structure_dir))

structures.head()
atom_count = structures.groupby('atom').size()

plt.bar(atom_count.index, atom_count.values)

plt.title('atom count');
def view_molecule(structures=structures, name=None):

    """

    Modified from:

    https://www.kaggle.com/borisdee/how-to-easy-visualization-of-molecules

    """

    if name is None:

        name = np.random.choice(structures.molecule_name.unique())

    print(f'molecule-name: {name}')

    

    molecule_df = structures.query('molecule_name == @name')

    molecule    = Atoms(positions=molecule_df[['x', 'y', 'z']].values,

                        symbols=molecule_df.atom.values)

    

    return ase.visualize.view(molecule, viewer="x3d")





view_molecule(name=None)



atom_counts = structures.groupby('molecule_name').size()

print(f'minimum number of atoms per molecule: {atom_counts.min()}')

print(f'maximum number of atoms per molecule: {atom_counts.max()}')

atom_counts.hist(bins=atom_counts.max())

plt.title('number of atoms per molecule');

train_structures = structures[structures.molecule_name.isin(set(train.molecule_name))]

print(f'number of traininig-structures: {len(train_structures.molecule_name.unique())}')

train_counts = train_structures.groupby('molecule_name').size()

train_counts.hist()

plt.title('training: number of atoms per molecule')

plt.show()



test_structures = structures[structures.molecule_name.isin(set(test.molecule_name))]

print(f'number of test-structures: {len(test_structures.molecule_name.unique())}')

test_structures.groupby('molecule_name').size().hist()

plt.title('test: number of atoms per molecule');
train_j_counts = train.groupby('molecule_name').size()

print(f'min and max number of j-couplings: {min(train_j_counts.values)}, {max(train_j_counts.values)}')

train_j_counts.hist()

plt.title('training-set: number of labelled interactions per molecule')

plt.show()



test.groupby('molecule_name').size().hist()

plt.title('test-set: number of labelled interactions per molecule');
for sc_type, type_df in train.groupby('type'):

    counts = type_df.groupby('molecule_name').size()

    counts.hist()

    plt.title(f'{sc_type}: number of labelled interactions')

    plt.show()
sc_types = train.type.unique()

print(f'{len(sc_types)} scalar coupling types:\n{sorted(sc_types)}')

print(f'\nbonds-apart: {set(int(x[0]) for x in sc_types)}')

print(f'\nsc-interactions: {set(x[2:4] for x in sc_types)}')
type_count = train.groupby('type').size() / len(train)

plt.bar(type_count.index, type_count.values)

plt.title('training: number of interactions per type')

plt.ylabel('fraction of interactions')

plt.xlabel('scalar coupling type')

plt.show()



type_count = test.groupby('type').size() / len(test)

plt.bar(type_count.index, type_count.values)

plt.ylabel('fraction of interactions')

plt.xlabel('scalar coupling type')

plt.title('test: number of interactions per type');
def add_sc_type_features(df):

    df['bonds']    = df.type.map(lambda x: int(x[0]))

    df['atom_pair'] = df.type.map(lambda x: x[2:4])

    return df





train = add_sc_type_features(train)

test  = add_sc_type_features(test)





bonds_counts = train.bonds.value_counts()

plt.bar(bonds_counts.index, bonds_counts.values)

plt.xlabel('bonds apart')

plt.ylabel('number of interactions')

plt.title('train: bonds between sc-atoms')

plt.show()



order_counts = test.bonds.value_counts()

plt.bar(order_counts.index, order_counts.values)

plt.xlabel('bonds apart')

plt.ylabel('number of interactions')

plt.title('test: bonds between sc-atoms')

plt.show()
atom_pair_types = train.atom_pair.value_counts()

plt.bar(atom_pair_types.index, atom_pair_types.values)

plt.xlabel('atom-pair')

plt.ylabel('number of interactions')

plt.title('train: sc atom-pairs')

plt.show()



atom_pair_types = train.atom_pair.value_counts()

plt.bar(atom_pair_types.index, atom_pair_types.values)

plt.xlabel('atom-pair')

plt.ylabel('number of interactions')

plt.title('test: bonds sc atom-pairs')

plt.show()
sns.boxplot(data=train, x='type', y='scalar_coupling_constant', fliersize=0.5)

plt.show();



sns.boxplot(data=train, x='bonds', y='scalar_coupling_constant', fliersize=0.5)

plt.show();



sns.boxplot(data=train, x='atom_pair', y='scalar_coupling_constant', fliersize=0.5)

plt.show();
train.scalar_coupling_constant.hist()

plt.title('(training) target-variable distribution');
dipole = pd.read_csv(join(DATA_DIR, 'dipole_moments.csv'))

assert not set(dipole.molecule_name).intersection(test.molecule_name)



display(dipole.head())

plt.hist(dipole[['X', 'Y', 'Z']].values.ravel())

plt.title('diple moments distribution');
pot_energy = pd.read_csv(join(DATA_DIR, 'potential_energy.csv'))

assert not set(pot_energy.molecule_name).intersection(test.molecule_name)



display(pot_energy.head())

pot_energy.potential_energy.hist()

plt.title('potential energy distribution');
mag_shield = pd.read_csv(join(DATA_DIR, 'magnetic_shielding_tensors.csv'))

assert not set(mag_shield.molecule_name).intersection(test.molecule_name)



display(mag_shield.head())

plt.hist(mag_shield.drop(['molecule_name', 'atom_index'], axis=1).values.ravel())

plt.title('magnetic shield distribution');
mulliken = pd.read_csv(join(DATA_DIR, 'mulliken_charges.csv'))

assert not set(mulliken.molecule_name).intersection(test.molecule_name)



display(mulliken.head())

mulliken.mulliken_charge.hist()

plt.title('mulliken charge distribution');
sc_contributions = pd.read_csv(join(DATA_DIR, 'scalar_coupling_contributions.csv'))

assert not set(sc_contributions.molecule_name).intersection(test.molecule_name)

display(sc_contributions.head())



for contribution in ('fc', 'sd', 'pso', 'dso'):

    sc_contributions[contribution].hist()

    plt.title(f'scalar coupling contribution: {contribution}')

    plt.show()
sc = pd.merge(sc_contributions, train, on=('molecule_name', 'atom_index_0', 'atom_index_1', 'type'))

assert len(sc) == len(train) == len(sc_contributions)



sc['contribution_sum'] = sc.fc + sc.sd + sc.pso + sc.dso

sc['difference'] = (sc.scalar_coupling_constant - sc.contribution_sum).map(abs)

print('maximal differnce between sum of scalar coupling contributions and\nthe target variable '

      f'scalar coupling: {sc.difference.max()}')

sc.head()
#Code in this cell is from this kernel:

#https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark



def map_atom_info(df, atom_idx, structures_df=structures):

    df = pd.merge(df, structures_df, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df





train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)



train.head()
# code in this cell is from:

# https://www.kaggle.com/artgor/brute-force-feature-engineering



train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
train.dist.hist()

plt.title('train: distances of target sc-interactions')

plt.xlabel('distance [Å]')

plt.ylabel('number of interactions (atom-pairs)');

plt.show()



test.dist.hist()

plt.title('test: distances of target sc-interactions')

plt.xlabel('distance [Å]')

plt.ylabel('number of interactions (atom-pairs)')

plt.show()



print(f'maximal sc-distance: {max(*train.dist, *test.dist):.2f}')
print('training-set:')

for n_bonds, df in train.groupby('bonds'):

    df.dist.hist()

    plt.title(f'train: distances of {n_bonds}-bond sc')

    plt.xlabel('distance [Å]')

    plt.ylabel('number of interactions (atom-pairs)');

    plt.show()

    

print('test-set:')

for n_bonds, df in test.groupby('bonds'):

    df.dist.hist()

    plt.title(f'test: distances of {n_bonds}-bond sc')

    plt.xlabel('distance [Å]')

    plt.ylabel('number of interactions (atom-pairs)');

    plt.show()
def get_dist_matrix(df_structures_idx, molecule, bug_fixed=True):

    """

    Function from:

    https://www.kaggle.com/cpmpml/ultra-fast-distance-matrix-computation

    """

    df_temp = df_structures_idx.loc[molecule]

    locs = df_temp[['x','y','z']].values

    num_atoms = len(locs)

    loc_tile = np.tile(locs.T, (num_atoms,1,1))

    

    # bug fix: first: sum up the squares, THEN take the square root

    if bug_fixed:

        return np.sqrt( ((loc_tile - loc_tile.T)**2).sum(axis=1) )

    else:

        return np.sqrt( (loc_tile - loc_tile.T)**2 ).sum(axis=1)





mol_name2dist_matrix = {}

structures.index = structures.molecule_name



for mol_name in tqdm_notebook(structures.molecule_name.unique()):

    mol_name2dist_matrix[mol_name] = get_dist_matrix(structures, mol_name)
max_dists = [matrix.max() for matrix in mol_name2dist_matrix.values()]

plt.hist(max_dists)

plt.title('Maximum atom-atom distance within molecule')

plt.ylabel('number of molecules')

plt.xlabel('distance [Å]')

plt.show()



min_dist = min(matrix[matrix != 0].min() for matrix in mol_name2dist_matrix.values())

print(f'minimum atom-to-atom position (over all molecules): {min_dist:.4f}')
# distance unit-test:

structures_idx = structures.set_index('molecule_name')

molname = 'dsgdb9nsd_000001'



mdf = structures_idx.loc[molname]



#dist_matrix_bug   = get_dist_matrix(structures_idx, molname, bug_fixed=False)

dist_matrix_fixed = get_dist_matrix(structures_idx, molname)



a = mdf.iloc[0][['x', 'y', 'z']].values

b = mdf.iloc[1][['x', 'y', 'z']].values

display(mdf.head(2))



print('scipy-euclidian: ', euclidean(a, b))

print('numpy-norm: ', np.linalg.norm(a - b))

print('generic: ', np.sqrt(((a - b)**2).sum()))

#print("CPMP's calculation (bug): ", dist_matrix_bug[0, 1])

print("CPMP's calculation (fixed): ", dist_matrix_fixed[0, 1])