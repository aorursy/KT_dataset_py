import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/train.csv')

df_structure = pd.read_csv('../input/structures.csv')

df_test = pd.read_csv('../input/test.csv')

df_mst = pd.read_csv('../input/magnetic_shielding_tensors.csv')

df_mc = pd.read_csv('../input/mulliken_charges.csv')

df_dp = pd.read_csv('../input/dipole_moments.csv')

df_pe = pd.read_csv('../input/potential_energy.csv')

df_contrib = pd.read_csv('../input/scalar_coupling_contributions.csv')
atoms = sorted(df_structure['atom'].unique())

mst_elems = sorted([c for c in df_mst.columns if c not in ['atom_index', 'molecule_name']])

dims = ['x', 'y', 'z']
atom_counts = df_structure.pivot_table(index='molecule_name', columns='atom', values='atom_index', aggfunc='count', fill_value=0)



fig, axs = plt.subplots(1, 5, figsize=(15, 3))

for c, ax in zip(atom_counts.columns, axs):

    ax.hist(atom_counts[c], bins=np.arange(atom_counts[c].max()) + 1)

    ax.set_title(c)
num_atoms = atom_counts.sum(axis=1)

plt.figure(figsize=(7,7))

num_atoms.hist(bins=np.arange(num_atoms.max()) + 1)

plt.title('Atom Count Histogram')

plt.show()
fig, axs = plt.subplots(5, 3, figsize=(15, 15))

for i, a in enumerate(atoms):

    axs[i, 0].set_ylabel(a, rotation=0, size='large')

    df_atom = df_structure[df_structure['atom'] == a]

    for j, d in enumerate(['x', 'y', 'z']):

        axs[i, j].hist(df_atom[d], bins=50)

for j, d in enumerate(['x', 'y', 'z']):

    axs[0, j].set_title(d)

plt.tight_layout()

plt.show()
pd.concat([df_structure[d].value_counts().value_counts() for d in dims], axis=1).fillna(0).astype(int)
df_train.head()
plt.figure(figsize=(9,9))

df_train['scalar_coupling_constant'].hist(bins=100)

plt.title('Scalar Coupline Constant Distribution')

plt.show()
plt.figure(figsize=(9,9))

df_train.boxplot('scalar_coupling_constant', by='type', ax=plt.gca())

plt.show()
contribs = ['fc', 'sd', 'pso', 'dso']
df_contrib.head()
df_contrib[contribs].hist(bins=100, figsize=(9, 9))

plt.suptitle('Scalar Coupling Contributions Distribution')

plt.show()
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

for i, c in enumerate(contribs):

    df_contrib.boxplot(c, by='type', ax=axs[i])

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(14, 7))



atom_index_dif_train = (df_train['atom_index_0'] - df_train['atom_index_1'])

atom_index_dif_train.hist(bins=np.arange(atom_index_dif_train.min(), atom_index_dif_train.max() + 1), ax=axs[0])



atom_index_dif_test = (df_test['atom_index_0'] - df_test['atom_index_1'])

atom_index_dif_test.hist(bins=np.arange(atom_index_dif_test.min(), atom_index_dif_test.max() + 1), ax=axs[1])



axs[0].set_title('Train')

axs[1].set_title('Test')

plt.suptitle('atom_index_0 - atom_index_1 distribution')



plt.show()
fig, axs = plt.subplots(1, 2, figsize=(14,7))

df_train.boxplot('scalar_coupling_constant', by='atom_index_0', ax=axs[0])

df_train.boxplot('scalar_coupling_constant', by='atom_index_1', ax=axs[1])

plt.title('Scalar Coupling Constant boxplots by atom_index')

plt.show()
df_train['index_0_gt_1'] = df_train['atom_index_0'] > df_train['atom_index_1']

df_test['index_0_gt_1'] = df_test['atom_index_0'] > df_test['atom_index_1']
df_train.hist('scalar_coupling_constant', by='index_0_gt_1', figsize=(14, 7), bins=100)

plt.suptitle('Scalar Couplig Constant distribution if atom_index_0 > atom_index_1')

plt.show()
df_train.pivot_table(index='type', columns='index_0_gt_1', aggfunc='count', values='id', fill_value=0)
df_test.pivot_table(index='type', columns='index_0_gt_1', aggfunc='count', values='id', fill_value=0)
fig, axs = plt.subplots(8, 2, figsize=(6, 24), sharex=True)

for t_idx, t in enumerate(df_train['type'].unique()):

    axs[t_idx, 0].set_ylabel(t, rotation=0, size='large')

    for tf_idx, tf in enumerate([True, False]):

        df_train[np.logical_and(

            df_train['type'] == t,

            df_train['index_0_gt_1'] == tf)]['scalar_coupling_constant'].hist(

            bins=100, density=True, ax=axs[t_idx, tf_idx])

axs[0, 0].set_title('atom_index_0 > atom_index_1')

axs[0, 1].set_title('atom_index_1 > atom_index_0')

plt.suptitle('Scalar Coupling Constant by Type and Funny Business')

plt.tight_layout()

plt.show()
df_mst.head()
df_mst[mst_elems].hist(bins=100, figsize=(9, 9))

plt.show()
mst_repeats = pd.concat([df_mst[c].value_counts().value_counts() for c in mst_elems], axis=1).fillna(0).astype(int)

mst_repeats.T
df_mc.head()
plt.figure()

df_mc['mulliken_charge'].hist(bins=100)
atom_mc = df_mc.merge(df_structure, how='left', on=['molecule_name', 'atom_index'])
fig, axs = plt.subplots(1, 5, figsize=(25, 5))

for c, ax in zip(atom_counts.columns, axs):

    df_atom = atom_mc[atom_mc['atom'] == c]

    ax.hist(df_atom['mulliken_charge'], bins=100)

    ax.set_title(c)
mc_repeats = atom_mc['mulliken_charge'].value_counts().value_counts()

pd.DataFrame(mc_repeats).T
df_dp.head()
df_dp[['X', 'Y', 'Z']].hist(bins=200, figsize=(21, 7), layout=(1,3))

plt.suptitle('Dipole XYZ Component Distributions')

plt.show()
pd.DataFrame((df_dp[['X', 'Y', 'Z']] == 0.0).sum(axis=1).value_counts(), columns=['Zero Components'])
fig, axs = plt.subplots(1, 3, figsize=(21, 7))

axs[0].scatter(df_dp['X'], df_dp['Y'], alpha=0.01)

axs[1].scatter(df_dp['X'], df_dp['Z'], alpha=0.01)

axs[2].scatter(df_dp['Y'], df_dp['Z'], alpha=0.01)

axs[0].set_title('XY')

axs[1].set_title('XZ')

axs[2].set_title('YZ')

plt.title('Dipole Moment Components Scattered')

plt.show()
df_dp[['X', 'Y', 'Z']].apply(np.linalg.norm, axis='columns').hist(bins=100, figsize=(7,7))

plt.title('L2 Norm of Dipole Moment Distribution')

plt.show()
pd.concat([df_dp[d].value_counts().value_counts() for d in ['X', 'Y', 'Z']], axis=1).fillna(0).astype(int).T
df_pe['potential_energy'].hist(bins=500, figsize=(15,5))

plt.title('Potential Energy Distribution')

plt.show()
pd.DataFrame(df_pe['potential_energy'].value_counts().value_counts()).T
fig, axs = plt.subplots(2, 2, figsize=(12,12))

df_train['atom_index_0'].hist(bins=np.arange(df_train['atom_index_0'].max()+1), ax=axs[0, 0])

df_train['atom_index_0'].hist(bins=np.arange(df_train['atom_index_1'].max()+1), ax=axs[0, 1])

df_test['atom_index_0'].hist(bins=np.arange(df_test['atom_index_0'].max()+1), ax=axs[1, 0])

df_test['atom_index_0'].hist(bins=np.arange(df_test['atom_index_1'].max()+1), ax=axs[1, 1])

axs[0, 0].set_ylabel('Train', rotation=0, size='large')

axs[1, 0].set_ylabel('Test', rotation=0, size='large')

axs[0, 0].set_title('atom_index_0')

axs[0, 1].set_title('atom_index_1')

plt.suptitle('Atom Index Distribution')

plt.show()
num_atoms_test = df_test.merge(pd.DataFrame(num_atoms, columns=['num_atoms']), on='molecule_name', how='left')

num_atoms_train = df_train.merge(pd.DataFrame(num_atoms, columns=['num_atoms']), on='molecule_name', how='left')
plt.figure()

plt.hist(num_atoms_train['num_atoms'], bins=np.arange(30), label='train', density=True, alpha=0.5)

plt.hist(num_atoms_test['num_atoms'], bins=np.arange(30), label='test', density=True, alpha=0.5)

plt.legend()

plt.title('Atoms per Molcule Distribution')

plt.show()
type_distribution = pd.concat([

    df_train['type'].value_counts() / len(df_train),

    df_test['type'].value_counts() / len(df_test)

], axis=1)

type_distribution.columns = ['train_type_distribution', 'test_type_distribution']

type_distribution
print('{} rows per molecule in train.'.format(len(df_train) / len(df_train['molecule_name'].unique())))

print('{} rows per molecule in test.'.format(len(df_test) / len(df_test['molecule_name'].unique())))
num_values_train = df_train.groupby('molecule_name')['atom_index_0'].apply(len)

df_atoms_values_train = pd.DataFrame(num_values_train)

df_atoms_values_train = df_atoms_values_train.merge(pd.DataFrame(num_atoms), left_index=True, right_index=True)

df_atoms_values_train.columns = ['num_values', 'num_atoms']



num_values_test = df_test.groupby('molecule_name')['atom_index_0'].apply(len)

df_atoms_values_test = pd.DataFrame(num_values_test)

df_atoms_values_test = df_atoms_values_test.merge(pd.DataFrame(num_atoms), left_index=True, right_index=True)

df_atoms_values_test.columns = ['num_values', 'num_atoms']



fig, axs = plt.subplots(1, 2, figsize=(12, 6))

(df_atoms_values_train['num_values'] / df_atoms_values_train['num_atoms']).hist(bins=50, ax=axs[0])

(df_atoms_values_test['num_values'] / df_atoms_values_test['num_atoms']).hist(bins=50, ax=axs[1])

axs[0].set_title('Train')

axs[1].set_title('Test')

plt.suptitle('CSV Rows Per Atom per Molecule')

plt.show()