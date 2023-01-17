import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/predicting-molecular-properties-datamerging/train.csv")

test = pd.read_csv("../input/predicting-molecular-properties-datamerging/test.csv")

structures = pd.read_csv("../input/champs-scalar-coupling/structures.csv")
train.drop(['X', 'Y', 'Z', 'potential_energy','XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0',

       'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',

       'mulliken_charge_atom_0','XX_atom_1', 'YX_atom_1', 'ZX_atom_1',

       'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1',

       'ZZ_atom_1', 'mulliken_charge_atom_1'], axis=1, inplace=True)
train.head(10)
test.head(10)
structures.head(10)
total_atoms_per_molecule = structures[['molecule_name']]

total_atoms_per_molecule['atoms_in_molecule'] = 1

total_atoms_per_molecule = total_atoms_per_molecule.groupby('molecule_name').sum()

total_atoms_per_molecule.head(10)
element_wise_atoms_per_molecule = structures[['molecule_name', 'atom']]

element_wise_atoms_per_molecule['count'] = 1

element_wise_atoms_per_molecule = element_wise_atoms_per_molecule.groupby(['molecule_name', 'atom']).sum()



atom_table = pd.pivot_table(element_wise_atoms_per_molecule, values='count', index=['molecule_name'],

                      columns=['atom']).fillna(0)

atom_table.head(10)
def get_cout_by_type(data):

    type_wise_bonds_per_molecule = data[['molecule_name', 'type']]

    type_wise_bonds_per_molecule['count'] = 1

    type_wise_bonds_per_molecule = type_wise_bonds_per_molecule.groupby(['molecule_name', 'type']).sum()   

    type_table = pd.pivot_table(type_wise_bonds_per_molecule, values='count', index=['molecule_name'],

                                columns=['type']).fillna(0)

    return type_table
def calculate_distance(data):

    data['x_dist'] = abs(data['x_atom_0'] - data['x_atom_1'])

    data['y_dist'] = abs(data['y_atom_0'] - data['y_atom_1'])

    data['z_dist'] = abs(data['z_atom_0'] - data['z_atom_1'])

    data['2_atom_dist'] = np.sqrt(data['x_dist']**2 + data['y_dist']**2 + data['z_dist']**2)
calculate_distance(train)

calculate_distance(test)
def calculate_mean_dist_mol_type(data):

    mean_dist_molecule_type = data[['molecule_name', 'type', '2_atom_dist']]

    mean_dist_molecule_type = mean_dist_molecule_type.groupby(['molecule_name', 'type']).mean()

    mean_dist_molecule_type.columns = ['mean_dist_for_mol_type']

#     print(mean_dist_molecule.head(20))

    return mean_dist_molecule_type
def calculate_mean_dist_mol(data):

    mean_dist_molecule = data[['molecule_name', '2_atom_dist']]

    mean_dist_molecule = mean_dist_molecule.groupby(['molecule_name']).mean()

    mean_dist_molecule.columns = ['mean_dist_for_mol']

#     print(mean_dist_molecule.head(20))

    return mean_dist_molecule
def calculate_mean_dist_type(data):

    mean_dist_type = data[['type', '2_atom_dist']]

    mean_dist_type = mean_dist_type.groupby(['type']).mean()

    mean_dist_type.columns = ['mean_dist_for_type']

#     print(mean_dist_type.head(20))

    return mean_dist_type
def calculate_max_dist_mol_type(data):

    max_dist_molecule_type = data[['molecule_name', 'type', '2_atom_dist']]

    max_dist_molecule_type = max_dist_molecule_type.groupby(['molecule_name', 'type']).max()

    max_dist_molecule_type.columns = ['max_dist_for_mol_type']

#     print(max_dist_molecule.head(20))

    return max_dist_molecule_type
def calculate_max_dist_mol(data):

    max_dist_molecule = data[['molecule_name', '2_atom_dist']]

    max_dist_molecule = max_dist_molecule.groupby(['molecule_name']).max()

    max_dist_molecule.columns = ['max_dist_for_mol']

#     print(max_dist_molecule.head(20))

    return max_dist_molecule
def calculate_max_dist_type(data):

    max_dist_type = data[['type', '2_atom_dist']]

    max_dist_type = max_dist_type.groupby(['type']).max()

    max_dist_type.columns = ['max_dist_for_type']

#     print(max_dist_type.head(20))

    return max_dist_type
def calculate_min_dist_mol_type(data):

    min_dist_molecule_type = data[['molecule_name', 'type', '2_atom_dist']]

    min_dist_molecule_type = min_dist_molecule_type.groupby(['molecule_name', 'type']).min()

    min_dist_molecule_type.columns = ['min_dist_for_mol_type']

#     print(min_dist_molecule.head(20))

    return min_dist_molecule_type
def calculate_min_dist_mol(data):

    min_dist_molecule = data[['molecule_name', '2_atom_dist']]

    min_dist_molecule = min_dist_molecule.groupby(['molecule_name']).min()

    min_dist_molecule.columns = ['min_dist_for_mol']

#     print(min_dist_molecule.head(20))

    return min_dist_molecule
def calculate_min_dist_type(data):

    min_dist_type = data[['type', '2_atom_dist']]

    min_dist_type = min_dist_type.groupby(['type']).min()

    min_dist_type.columns = ['min_dist_for_type']

#     print(min_dist_type.head(20))

    return min_dist_type
train = train.merge(total_atoms_per_molecule, on='molecule_name')

train = train.merge(atom_table, on='molecule_name')

train = train.merge(get_cout_by_type(train), on='molecule_name')



train = train.merge(calculate_mean_dist_mol(train), on='molecule_name')

train = train.merge(calculate_max_dist_mol(train), on='molecule_name')

train = train.merge(calculate_min_dist_mol(train), on='molecule_name')



train = train.merge(calculate_mean_dist_type(train), on='type', how='left')

train = train.merge(calculate_max_dist_type(train), on='type', how='left')

train = train.merge(calculate_min_dist_type(train), on='type', how='left')



train = train.merge(calculate_mean_dist_mol_type(train), on=['molecule_name', 'type'])

train = train.merge(calculate_max_dist_mol_type(train), on=['molecule_name', 'type'])

train = train.merge(calculate_min_dist_mol_type(train), on=['molecule_name', 'type'])



test = test.merge(total_atoms_per_molecule, on='molecule_name')

test = test.merge(atom_table, on='molecule_name')

test = test.merge(get_cout_by_type(test), on='molecule_name')



test = test.merge(calculate_mean_dist_mol(test), on='molecule_name')

test = test.merge(calculate_max_dist_mol(test), on='molecule_name')

test = test.merge(calculate_min_dist_mol(test), on='molecule_name')



test = test.merge(calculate_mean_dist_type(test), on='type', how='left')

test = test.merge(calculate_max_dist_type(test), on='type', how='left')

test = test.merge(calculate_min_dist_type(test), on='type', how='left')



test = test.merge(calculate_mean_dist_mol_type(test), on=['molecule_name', 'type'])

test = test.merge(calculate_max_dist_mol_type(test), on=['molecule_name', 'type'])

test = test.merge(calculate_min_dist_mol_type(test), on=['molecule_name', 'type'])
train
def add_features(data):

    data['dist-mean_dist_mol'] = data['2_atom_dist'] - data['mean_dist_for_mol']

    data['dist-min_dist_mol'] = data['2_atom_dist'] - data['min_dist_for_mol']

    data['max_dist_mol-dist'] = data['max_dist_for_mol'] - data['2_atom_dist']

    

    data['dist-mean_dist_type'] = data['2_atom_dist'] - data['mean_dist_for_type']

    data['dist-min_dist_type'] = data['2_atom_dist'] - data['min_dist_for_type']

    data['max_dist_type-dist'] = data['max_dist_for_type'] - data['2_atom_dist']

    

    data['dist-mean_dist_mol_type'] = data['2_atom_dist'] - data['mean_dist_for_mol_type']

    data['dist-min_dist_mol_type'] = data['2_atom_dist'] - data['min_dist_for_mol_type']

    data['max_dist_mol-dist_type'] = data['max_dist_for_mol_type'] - data['2_atom_dist']
add_features(train)

add_features(test)
types = train[['molecule_name', 'type']]

types = types.groupby('type').count().sort_values('molecule_name', ascending=False)



plt.figure(figsize=(10,8))

sns.barplot(types.index, types['molecule_name'])

plt.title('Coupling type wise number of pairs')

plt.ylabel('Number of pairs')

plt.show()
# coupling = train[['type', 'scalar_coupling_constant']]

# plt.figure(figsize=(20,12))

# sns.violinplot(x='type', y='scalar_coupling_constant', data=coupling)

# plt.ylabel('scalar_coupling_constant')

# plt.title('Distribution of scalar coupling constant')

# plt.show()
# plt.figure(figsize=(20,12))

# sns.relplot(x='2_atom_dist', y='scalar_coupling_constant', hue='type', data=train)

# plt.xlabel('Distance between 2 atoms')

# plt.ylabel('scalar coupling constant')

# plt.show()
# avg_dist = train[['type', '2_atom_dist', 'scalar_coupling_constant']]

# avg_dist['count'] = 1

# avg_dist = avg_dist.groupby('type').agg({'2_atom_dist': np.mean,

#                                        'scalar_coupling_constant': np.mean,

#                                        'count':np.sum})

# sns.relplot(x='2_atom_dist', y='scalar_coupling_constant', hue=avg_dist.index, 

#             size='count', sizes=(50,1000), data=avg_dist)

# plt.xlabel('Avg. distance between 2 atoms')

# plt.ylabel('Avg. scalar coupling constant')

# plt.show()
train.columns
columns = ['type',

           '2_atom_dist', 'dist-mean_dist_mol',

       'dist-min_dist_mol', 'max_dist_mol-dist', 'dist-mean_dist_type',

       'dist-min_dist_type', 'max_dist_type-dist',

           'fc', 'sd', 'pso', 'dso',

           'scalar_coupling_constant']



df_1 = train[columns]
def draw_heatmap(df):

    corr_mx = df.corr()

    plt.figure(figsize=(12,8))

    sns.heatmap(corr_mx, annot=True, vmin=-1, vmax=1, center=0)

    plt.show()
df = df_1[df_1['type'] == '1JHC']

draw_heatmap(df)
df = df_1[df_1['type'] == '1JHN']

draw_heatmap(df)
df = df_1[df_1['type'] == '2JHC']

draw_heatmap(df)
df = df_1[df_1['type'] == '2JHN']

draw_heatmap(df)
df = df_1[df_1['type'] == '2JHH']

draw_heatmap(df)
df = df_1[df_1['type'] == '3JHC']

draw_heatmap(df)
df = df_1[df_1['type'] == '3JHN']

draw_heatmap(df)
df = df_1[df_1['type'] == '3JHH']

draw_heatmap(df)
def to_onehot_type(data):

    encoder = OneHotEncoder(sparse=False)

    onehot_type = encoder.fit_transform(data['type'].values.reshape(len(data),1))

    df_onehot = pd.DataFrame(onehot_type, columns=['type_'+t for t in encoder.categories_[0]])

    data = pd.concat([data, df_onehot], axis=1)

    return data
train = to_onehot_type(train)

test = to_onehot_type(test)
train = train[['id', 'type',

               'type_1JHC', 'type_1JHN', 'type_2JHC', 'type_2JHH', 

               'type_2JHN', 'type_3JHC', 'type_3JHH', 'type_3JHN', 

               'atoms_in_molecule', 

               'C', 'F', 'H', 'N', 'O', 

               '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN',

               '2_atom_dist', 

               'mean_dist_for_mol', 'max_dist_for_mol', 'min_dist_for_mol', 

               'mean_dist_for_type', 'max_dist_for_type', 'min_dist_for_type', 

               'dist-mean_dist_mol', 'dist-min_dist_mol', 'max_dist_mol-dist', 

               'dist-mean_dist_type', 'dist-min_dist_type', 'max_dist_type-dist',

               'mean_dist_for_mol_type', 'max_dist_for_mol_type', 'min_dist_for_mol_type',

               'dist-mean_dist_mol_type', 'dist-min_dist_mol_type', 'max_dist_mol-dist_type',

               'fc', 'sd', 'pso', 'dso', 

               'scalar_coupling_constant']]
test = test[['id', 'type',

             'type_1JHC', 'type_1JHN', 'type_2JHC', 'type_2JHH', 

             'type_2JHN', 'type_3JHC', 'type_3JHH', 'type_3JHN', 

             'atoms_in_molecule', 

             'C', 'F', 'H', 'N', 'O', 

             '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN',

             '2_atom_dist', 

             'mean_dist_for_mol', 'max_dist_for_mol', 'min_dist_for_mol', 

             'mean_dist_for_type', 'max_dist_for_type', 'min_dist_for_type', 

             'dist-mean_dist_mol', 'dist-min_dist_mol', 'max_dist_mol-dist', 

             'dist-mean_dist_type', 'dist-min_dist_type', 'max_dist_type-dist',

             'mean_dist_for_mol_type', 'max_dist_for_mol_type', 'min_dist_for_mol_type',

             'dist-mean_dist_mol_type', 'dist-min_dist_mol_type', 'max_dist_mol-dist_type']]
train.head(10)
test.head(10)
train.to_csv('train.csv', index=False)

test.to_csv('test.csv', index=False)