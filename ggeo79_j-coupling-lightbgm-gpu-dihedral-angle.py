# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import cupy as cp

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder, StandardScaler

from scipy import sparse

from tqdm import tqdm_notebook as tqdm



sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', header=0)
df_train.head()
fig, ax = plt.subplots(figsize=(14, 12))

sns.boxplot(x='type', y='scalar_coupling_constant', data=df_train, ax=ax)

plt.show()
df_test = pd.read_csv('../input/test.csv', header=0)

df_test.head()
structures = pd.read_csv('../input/structures.csv')
structures.head()
atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor



fudge_factor = 0.05

atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}



electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}



#structures = pd.read_csv(structures, dtype={'atom_index':np.int8})



atoms = structures['atom'].values

atoms_en = [electronegativity[x] for x in tqdm(atoms)]

atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]



structures['EN'] = atoms_en

structures['rad'] = atoms_rad
structures.head()
i_atom = structures['atom_index'].values

p = structures[['x', 'y', 'z']].values

p_compare = p

m = structures['molecule_name'].values

m_compare = m

r = structures['rad'].values

r_compare = r



source_row = np.arange(len(structures))

max_atoms = 28



bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)

bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)





for i in tqdm(range(max_atoms-1)):

    p_compare = np.roll(p_compare, -1, axis=0)

    m_compare = np.roll(m_compare, -1, axis=0)

    r_compare = np.roll(r_compare, -1, axis=0)

    

    mask = np.where(m == m_compare, 1, 0) #Are we still comparing atoms in the same molecule?

    dists = np.linalg.norm(p - p_compare, axis=1) * mask

    r_bond = r + r_compare

    

    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

    

    source_row = source_row

    target_row = source_row + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) #If invalid target, write to dummy row

    

    source_atom = i_atom

    target_atom = i_atom + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) #If invalid target, write to dummy col

    

    bonds[(source_row, target_atom)] = bond

    bonds[(target_row, source_atom)] = bond

    bond_dists[(source_row, target_atom)] = dists

    bond_dists[(target_row, source_atom)] = dists



bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row

bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col

bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row

bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col





bonds_numeric = [[i for i,x in enumerate(row) if x] for row in tqdm(bonds)]

bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(tqdm(bond_dists))]

bond_lengths_mean = [ np.mean(x) for x in bond_lengths]

n_bonds = [len(x) for x in bonds_numeric]



bond_data = {'n_bonds':n_bonds, 'bond_lengths_mean': bond_lengths_mean }

bond_df = pd.DataFrame(bond_data)

structures = structures.join(bond_df)
structures.head()
# https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python

def dihedral_angle(data): 

        

    vals = np.array(data[:, 3:6], dtype=np.float64)

    mol_names = np.array(data[:, 0], dtype=np.str)

 

    result = np.zeros((data.shape[0], 2), dtype=object)

    # use every 4 rows to compute the dihedral angle

    for idx in range(0, vals.shape[0] - 4, 4):



        a0 = vals[idx]

        a1 = vals[idx + 1]

        a2 = vals[idx + 2]

        a3 = vals[idx + 3]

        

        b0 = a0 - a1

        b1 = a2 - a1

        b2 = a3 - a2

        

        # normalize b1 so that it does not influence magnitude of vector

        # rejections that come next

        b1 /= np.linalg.norm(b1)

    

        # vector rejections

        # v = projection of b0 onto plane perpendicular to b1

        #   = b0 minus component that aligns with b1

        # w = projection of b2 onto plane perpendicular to b1

        #   = b2 minus component that aligns with b1



        v = b0 - np.dot(b0, b1) * b1

        w = b2 - np.dot(b2, b1) * b1



        # angle between v and w in a plane is the torsion angle

        # v and w may not be normalized but that's fine since tan is y/x

        x = np.dot(v, w)

        y = np.dot(np.cross(b1, v), w)

       

        # We want all 4 first rows for every molecule to have the same value

        # (in order to have the same length as the dataframe)

        result[idx:idx + 4] = [mol_names[idx], np.degrees(np.arctan2(y, x))]

        

    return result
from datetime import datetime

startTime = datetime.now()

dihedral = dihedral_angle(structures[structures.groupby('molecule_name')['atom_index'].transform('count').ge(4)].groupby('molecule_name').head(4).values)

print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - startTime))
themap = {k:v for k, v in dihedral if k}
structures['dihedral'] = structures['molecule_name'].map(themap)
structures.head()
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
train = map_atom_info(df_train, 0)

train = map_atom_info(train, 1)
test = map_atom_info(df_test, 0)

test = map_atom_info(test, 1)
train.head()
# Euclidean Distance

def dist(a, b, ax=1):

    return cp.linalg.norm(a - b, axis=ax)
train_atom_0 = cp.asarray(train[['x_0', 'y_0', 'z_0']].values)

train_atom_1 = cp.asarray(train[['x_1', 'y_1', 'z_1']].values)



train['distance'] = dist(train_atom_1, train_atom_0).get()

train['dist_x'] = dist( cp.asarray(train[['x_0']].values),  cp.asarray(train[['x_1']].values)).get()

train['dist_y'] = dist( cp.asarray(train[['y_0']].values),  cp.asarray(train[['y_1']].values)).get()

train['dist_z'] = dist( cp.asarray(train[['z_0']].values),  cp.asarray(train[['z_1']].values)).get()
test_atom_0 = cp.asarray(test[['x_0', 'y_0', 'z_0']].values)

test_atom_1 = cp.asarray(test[['x_1', 'y_1', 'z_1']].values)



test['distance'] = dist(test_atom_1, test_atom_0).get()

test['dist_x'] = dist( cp.asarray(test[['x_0']].values),  cp.asarray(test[['x_1']].values)).get()

test['dist_y'] = dist( cp.asarray(test[['y_0']].values),  cp.asarray(test[['y_1']].values)).get()

test['dist_z'] = dist( cp.asarray(test[['z_0']].values),  cp.asarray(test[['z_1']].values)).get()
train.head()
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
def create_features(df):

    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')

    df['molecule_dist_mean'] = df.groupby('molecule_name')['distance'].transform('mean')

    df['molecule_dist_min'] = df.groupby('molecule_name')['distance'].transform('min')

    df['molecule_dist_max'] = df.groupby('molecule_name')['distance'].transform('max')

    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')

    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')

    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']

    #df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']

    #df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')

    #df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')

    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('mean')

    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['distance']

    #df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['distance']

    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('max')

    #df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['distance']

    #df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['distance']

    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min')

    #df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['distance']

    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['distance']

    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('std')

    #df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['distance']

    #df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['distance']

    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('mean')

    #df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['distance']

    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['distance']

    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('max')

    #df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['distance']

    #df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['distance']

    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('min')

    #df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['distance']

    #df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['distance']

    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('std')

    #df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['distance']

    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['distance']

    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['distance'].transform('mean')

    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['distance'].transform('min')

    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['distance']

    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['distance']

    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['distance'].transform('std')

    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['distance']

    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['distance'].transform('mean')

    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['distance']

    #df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['distance']

    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['distance'].transform('max')

    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['distance'].transform('min')

    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['distance'].transform('std')

    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['distance']



    df = reduce_mem_usage(df)

    return df
train = create_features(train)

test = create_features(test)
train.head()
for i in ['atom_0', 'atom_1', 'type']:

    class_le = LabelEncoder()   

    train[i] = class_le.fit_transform(train[i].values)

    test[i] = class_le.fit_transform(test[i].values)
cols = [



'id',

'atom_index_0', 

'atom_index_1', 

'type', 

'atom_0',

'EN_x',

'rad_x',

'n_bonds_x',

'bond_lengths_mean_x',

'x_0', 

'y_0', 

'z_0', 

'dihedral_x',

'atom_1',

'EN_y',

'rad_y',

'n_bonds_y',

'bond_lengths_mean_y',

'x_1', 

'y_1', 

'z_1', 

'dihedral_y',

'distance',

'dist_x', 

'dist_y', 

'dist_z',

'molecule_atom_index_0_dist_min',

'molecule_atom_index_0_dist_max',

'molecule_atom_index_1_dist_min',

'molecule_atom_index_0_dist_mean',

'molecule_atom_index_0_dist_std',

'molecule_atom_index_1_dist_std',

'molecule_atom_index_1_dist_max',

'molecule_atom_index_1_dist_mean',

#'molecule_atom_index_0_dist_max_diff',

#'molecule_atom_index_0_dist_max_div',

#'molecule_atom_index_0_dist_std_diff',

#'molecule_atom_index_0_dist_std_div',

'atom_0_couples_count',

'molecule_atom_index_0_dist_min_div',

#'molecule_atom_index_1_dist_std_diff',

#'molecule_atom_index_0_dist_mean_div',

'atom_1_couples_count',

'molecule_atom_index_0_dist_mean_diff',

'molecule_couples',

'molecule_dist_mean',

#'molecule_atom_index_1_dist_max_diff',

'molecule_atom_index_0_y_1_std',

#'molecule_atom_index_1_dist_mean_diff',

'molecule_atom_index_1_dist_std_div',

'molecule_atom_index_1_dist_mean_div',

#'molecule_atom_index_1_dist_min_diff',

#'molecule_atom_index_1_dist_min_div',

#'molecule_atom_index_1_dist_max_div',

'molecule_atom_index_0_z_1_std',

'molecule_type_dist_std_diff',

'molecule_atom_1_dist_min_diff',

'molecule_atom_index_0_x_1_std',

'molecule_dist_min',

#'molecule_atom_index_0_dist_min_diff',

'molecule_atom_index_0_y_1_mean_diff',

'molecule_type_dist_min',

'molecule_atom_1_dist_min_div',

'molecule_dist_max',

'molecule_atom_1_dist_std_diff',

'molecule_type_dist_max',

#'molecule_atom_index_0_y_1_max_diff',

'molecule_type_dist_mean_diff',

'molecule_atom_1_dist_mean',

#'molecule_atom_index_0_y_1_mean_div',

#'molecule_type_dist_mean_div'

]
def data(df):

    X_train, X_val, y_train, y_val  = train_test_split(df[cols].values,

                                                       df.loc[:, 'scalar_coupling_constant'].values,

                                                       test_size=0.2,

                                                       random_state=1340)

        

    return X_train, X_val, y_train, y_val
X_test = test[cols].values
num_boost_round = 4000

early_stopping_rounds = 200

verbose_eval = 200



X_train, X_val, y_train, y_val = data(train)
lgb_train = lgb.Dataset(X_train, y_train)

lgb_val = lgb.Dataset(X_val, y_val)

evals_result = {}



params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'mae',

            'learning_rate': 0.2,

            'num_leaves': 900, 

            'reg_alpha': 0.5, 

            'reg_lambda': 0.5, 

            'nthread': 4, 

            'device': 'cpu',

            'min_child_samples': 45

        }
model = lgb.train(params,

                  lgb_train,

                  num_boost_round=num_boost_round,

                  valid_sets=[lgb_val],

                  early_stopping_rounds=early_stopping_rounds, 

                  evals_result=evals_result, 

                  verbose_eval=verbose_eval)
#model.save_model('model.txt', num_iteration=model.best_iteration)
preds = model.predict(X_test, num_iteration=model.best_iteration)
def submit(predictions):

    submit = pd.read_csv('../input/sample_submission.csv')

    submit["scalar_coupling_constant"] = predictions

    submit.to_csv("submission.csv", index=False)
submit(preds)