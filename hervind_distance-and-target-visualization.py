import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.colors as colors

import matplotlib.cm as cmx

import seaborn as sns

sns.set()



import os

print(os.listdir("../input"))



from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor



train = pd.read_csv('../input/train.csv', index_col='id')

test = pd.read_csv('../input/test.csv', index_col='id')



structures = pd.read_csv('../input/structures.csv')



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



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
%%time

# This block is SPPED UP



train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist_speedup'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist_speedup'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
train.head(5)
train['same_atom'] = (train['atom_0'] == train['atom_1']).astype('int')
train_part = train.loc[:10000]
train.plot.scatter(x = 'dist_speedup', y = 'scalar_coupling_constant', c = 'same_atom',

                       cmap = 'summer', alpha = 0.4)

plt.show()
list_type = list(set(train['type']))

cNorm  = colors.Normalize(vmin=0, vmax=len(list_type))

cmap_plot = plt.get_cmap('Set1')

scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap_plot)



for i in range(len(list_type)):

    indx = train['type'] == list_type[i]

    plt.scatter(x = train.loc[indx, 'dist_speedup'], y = train.loc[indx, 'scalar_coupling_constant'],

               color = scalarMap.to_rgba(i), label = list_type[i], alpha = 0.4)

    

plt.xlabel('dist_speedup')

plt.ylabel('scalar_coupling_constant')

plt.legend(loc='upper right')

plt.show()