%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import os

import pandas as pd



DATA_DIR = '../input/'

print(os.listdir(DATA_DIR))
train_df = pd.read_csv(DATA_DIR + 'train.csv', index_col=0)

test_df = pd.read_csv(DATA_DIR + 'test.csv')

structures_df = pd.read_csv(DATA_DIR + 'structures.csv')

sc_contri_df = pd.read_csv(DATA_DIR + 'scalar_coupling_contributions.csv')
train_mol = train_df.molecule_name.unique()

test_mol =  test_df.molecule_name.unique()

structure_mol =  structures_df.molecule_name.unique()

print('Train unique molecules', len(train_mol))

print('Test unique molecules', len(test_mol))

print('Structures unique molecules', len(structure_mol))

print('Common unique molecules', len(set(train_mol).intersection(set(test_mol))))
print('1st atom index types', len(train_df.atom_index_0.unique()))

print('2nd atom index types', len(train_df.atom_index_1.unique()))

print('All atom indices', len(set(train_df.atom_index_0.unique()).union(train_df.atom_index_1.unique())))

print(set(train_df.atom_index_0.unique()).union(train_df.atom_index_1.unique()))

print('Atom types', structures_df.atom.unique())



print('Molecule type ', test_df['type'].unique())

print('Molecule types are same in train,test', set(train_df['type'].unique()) == set(test_df['type'].unique()))
for contri_col in ['fc','sd','pso','dso']:

    train_df[contri_col] = sc_contri_df[contri_col]
_, ax = plt.subplots(figsize=(10,10))

vc_train = train_df['type'].value_counts().to_frame('train')

vc_test =test_df['type'].value_counts().to_frame('test')

pd.concat([vc_train, vc_test],axis=1).plot(ax=ax)

ax.set_xticklabels(['0'] + vc_train.index.tolist(), fontsize=10)

ax.legend()

(sc_contri_df[['fc','sd','pso','dso']].sum(axis=1) - train_df.scalar_coupling_constant).describe()
sc_contri_df[['fc','sd','pso','dso']].describe().loc[['mean','std','min','max']]
_, ax = plt.subplots(figsize=(10,10))

sns.heatmap(sc_contri_df[['fc','sd','pso','dso']].corr(), annot=True, ax=ax)
temp_df = structures_df.groupby('atom')['atom_index'].apply(lambda x: x.unique())

df = pd.DataFrame(index=['C','F','H','N','O'], columns=list(range(29)))

df.index.name = 'atom'

df.columns.name ='atom_index'

df[:] = 0

df = df.stack().to_frame('present').reset_index(level=1)



for atom in set(df.index.values):

    df.loc[(df.index ==atom) &(df.atom_index.isin(temp_df.loc[atom])), 'present'] = 1

    

_, ax= plt.subplots(figsize=(10,5))

sns.heatmap(df.set_index('atom_index',append=True)['present'].unstack())

ax.set_title('Relationship between atom and atom_index')

del df, temp_df
Y_COL ='scalar_coupling_constant'
_, ax = plt.subplots(nrows=2, figsize=(10, 20))

sns.violinplot(x='atom_index_0', y=Y_COL, data=train_df, ax=ax[0])

sns.violinplot(x='atom_index_1', y=Y_COL, data=train_df, ax=ax[1])
sns.violinplot(x='type', y=Y_COL, data=train_df)
from sklearn.preprocessing import LabelEncoder

train_df['type_enc'] = LabelEncoder().fit_transform(train_df['type'])

train_df['idx1_type'] = train_df['atom_index_1'] + train_df['type_enc']*train_df['atom_index_1'].max()

_, ax = plt.subplots(figsize=(20,5))

sns.violinplot(x='idx1_type', y=Y_COL, data=train_df,ax=ax)
# Taken from https://www.kaggle.com/artgor/molecular-properties-eda-and-models

import networkx as nx

fig, ax = plt.subplots(figsize = (20, 12))

for i, t in enumerate(train_df['type'].unique()):

    train_df_type = train_df.loc[train_df['type'] == t]

    bad_atoms_0 = list(train_df_type['atom_index_0'].value_counts(normalize=True)[train_df_type['atom_index_0'].value_counts(normalize=True) < 0.01].index)

    bad_atoms_1 = list(train_df_type['atom_index_1'].value_counts(normalize=True)[train_df_type['atom_index_1'].value_counts(normalize=True) < 0.01].index)

    bad_atoms = list(set(bad_atoms_0 + bad_atoms_1))

    train_df_type = train_df_type.loc[(train_df_type['atom_index_0'].isin(bad_atoms_0) == False) & (train_df_type['atom_index_1'].isin(bad_atoms_1) == False)]

    G = nx.from_pandas_edgelist(train_df_type, 'atom_index_0', 'atom_index_1', ['scalar_coupling_constant'])

    plt.subplot(2, 4, i + 1);

    nx.draw(G, with_labels=True);

    plt.title(f'Graph for type {t}')
_, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,10))

sns.violinplot(x='atom_index_0', data=train_df, ax=ax[0,0])

ax[0,0].set_title('Train')

sns.violinplot(x='atom_index_1', data=train_df, ax=ax[1,0])

ax[1,0].set_title('Train')

sns.violinplot(x='atom_index_0', data=test_df, ax=ax[0,1])

ax[0,1].set_title('Test')

sns.violinplot(x='atom_index_1', data=test_df, ax=ax[1,1])

ax[1,1].set_title('Test')
def plot_molecule(mol_name, figsize=(10,10)):

    from mpl_toolkits.mplot3d import Axes3D

    temp_df = structures_df[structures_df.molecule_name == mol]



    marker_size = {'C':120,'H':10,'N':70,'F':90,'O':80}

    marker_color ={'C':'blue','H':'orange','N':'green','F':'black','O':'violet'}

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111, projection='3d')

    for atom in temp_df.atom.unique():

        dta = temp_df[temp_df.atom == atom]

        ax.scatter(dta.x, dta.y, dta.z, s=marker_size[atom], c=marker_color[atom],label=atom)

        for _, row in dta.iterrows():

            ax.text(row.x, row.y, row.z,row.atom_index,fontsize=20)

    ax.legend()

temp_df = structures_df.groupby('molecule_name')['atom_index'].agg({

    'sum':'sum',

    'count':'count'})

temp_df['all_present_sum'] = temp_df['count']*(temp_df['count'] -1)/2

assert(temp_df['all_present_sum'] == temp_df['sum']).all()
mol = 'dsgdb9nsd_121674'

plot_molecule(mol,figsize=(20,20))
temp_df = structures_df.molecule_name.value_counts()

print(temp_df.describe())

temp_df.hist()

del temp_df
import numpy as np

train_df['target_sign'] = (train_df['scalar_coupling_constant'] > 0).astype(np.int)
temp_df =pd.merge(train_df, structures_df, how='left',left_on=['molecule_name','atom_index_0'],

         right_on=['molecule_name','atom_index'])



temp_df =pd.merge(temp_df, structures_df, how='left',left_on=['molecule_name','atom_index_1'],

         right_on=['molecule_name','atom_index'])

print(temp_df.groupby('atom_y')['target_sign'].describe()[['mean','std','min','25%']])

del temp_df
_, ax = plt.subplots(figsize=(20,5))

sns.distplot(train_df.molecule_name.value_counts(),ax=ax,label='train')

sns.distplot(test_df.molecule_name.value_counts(),ax=ax, label='test')

ax.set_xlabel('Number of rows')

ax.legend()