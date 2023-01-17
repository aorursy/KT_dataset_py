import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pylab as plt

import seaborn as sns

from sklearn import metrics

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



print('train_df.shape',train_df.shape)

print('test_df.shape',test_df.shape)

print('train_df.columns',train_df.columns.values)


structures = pd.read_csv('../input/structures.csv')

print('structures.shape',structures.shape)

print('structures.columns',structures.columns.values)
print('train_df',train_df.shape)

display(train_df.head(2))

print('')



print('test_df',test_df.shape)

display(test_df.head(2))

print('')



print('structures',structures.shape)

display(structures.head(2))

print('')

plt.rcParams['font.size'] = 20

fig, ax = plt.subplots(2, 2, figsize=(20, 10))



train_df['scalar_coupling_constant'].plot(kind='hist',ax=ax.flat[0],  bins=1000, title='scalar_coupling_constant')

train_df['type'].value_counts().plot(kind='bar',ax=ax.flat[1], title='type')

train_df['atom_index_0'].plot(kind='hist',  bins=25, ax=ax.flat[2],title='atom_index_0')

train_df['atom_index_1'].plot(kind='hist', bins=25, ax=ax.flat[3],title='atom_index_1')

plt.tight_layout() 

plt.show()
def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



train_df = map_atom_info(train_df, 0)

train_df = map_atom_info(train_df, 1)



test_df = map_atom_info(test_df, 0)

test_df = map_atom_info(test_df, 1)



atom_count_dict = structures.groupby('molecule_name').count()['atom_index'].to_dict()



train_df['atom_count'] = train_df['molecule_name'].map(atom_count_dict)

test_df['atom_count'] = test_df['molecule_name'].map(atom_count_dict)
print('train_df.columns',train_df.columns.values)

train_df.head(3)
plt.rcParams['font.size'] = 20

fig, ax = plt.subplots(2, 3, figsize=(20, 10))

train_df['x_0'].plot(kind='hist', ax=ax.flat[0], bins=500, title='Atom_0_X')

train_df['y_0'].plot(kind='hist', ax=ax.flat[1], bins=500, title='Atom_0_Y')

train_df['z_0'].plot(kind='hist', ax=ax.flat[2], bins=500, title='Atom_0_Z')

train_df['x_1'].plot(kind='hist', ax=ax.flat[3], bins=500, title='Atom_1_X')

train_df['y_1'].plot(kind='hist', ax=ax.flat[4], bins=500, title='Atom_1_Y')

train_df['z_1'].plot(kind='hist', ax=ax.flat[5], bins=500, title='Atom_1_Z')

plt.tight_layout()

plt.show()
plt.rcParams['font.size'] = 20

fig, ax = plt.subplots(2, 3, figsize=(20, 10))

test_df['x_0'].plot(kind='hist', ax=ax.flat[0], bins=500, title='Atom_0_X')

test_df['y_0'].plot(kind='hist', ax=ax.flat[1], bins=500, title='Atom_0_Y')

test_df['z_0'].plot(kind='hist', ax=ax.flat[2], bins=500, title='Atom_0_Z')

test_df['x_1'].plot(kind='hist', ax=ax.flat[3], bins=500, title='Atom_1_X')

test_df['y_1'].plot(kind='hist', ax=ax.flat[4], bins=500, title='Atom_1_Y')

test_df['z_1'].plot(kind='hist', ax=ax.flat[5], bins=500, title='Atom_1_Z')

plt.tight_layout()

plt.show()
print('train atom_0:',train_df['atom_0'].value_counts())

print('test atom_0:',test_df['atom_0'].value_counts())



print('train atom_1:',train_df['atom_1'].value_counts())

print('test atom_1:',test_df['atom_1'].value_counts())



plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(1, 2, figsize=(5, 3))



train_df['atom_1'].value_counts().plot(kind='bar', ax=ax.flat[0], title='train atom_1')

test_df['atom_1'].value_counts().plot(kind='bar', ax=ax.flat[1],  title='test atom_1')

plt.tight_layout()

plt.show()
# Atomic species contained in the molecule

print('structures_atom:',structures['atom'].value_counts())

# Histogram of all atom positions



plt.rcParams['font.size'] = 10



fig, ax = plt.subplots(1, 3, figsize=(10, 3))

structures['x'].plot(kind='hist', ax=ax.flat[0], bins=500, title='Atom_X')

structures['y'].plot(kind='hist', ax=ax.flat[1], bins=500, title='Atom_Y')

structures['z'].plot(kind='hist', ax=ax.flat[2], bins=500, title='Atom_Z')



plt.tight_layout()

plt.show()
# Add distance information of atom0 and atom1 to train_df and test_df



train_p_0 = train_df[['x_0', 'y_0', 'z_0']].values

train_p_1 = train_df[['x_1', 'y_1', 'z_1']].values

test_p_0 = test_df[['x_0', 'y_0', 'z_0']].values

test_p_1 = test_df[['x_1', 'y_1', 'z_1']].values



train_df['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test_df['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
sns.pairplot(data=train_df.sample(5000), hue='type',

             vars=['scalar_coupling_constant',  'dist']).savefig('seaborn_pairplot_vars.png')