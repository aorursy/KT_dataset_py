# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from numpy.random import permutation

from sklearn import metrics

import lightgbm

from matplotlib import pyplot as plt

import math
train = pd.read_csv('../input/champs-scalar-coupling/train.csv')

test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

sub = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv')
# previously constructed features

train_dist = pd.read_csv('../input/distance-features/train_dist.csv')

test_dist = pd.read_csv('../input/distance-features/test_dist.csv')
# get xyz data for each atom

structures = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
# dipole moments of molecules

dipole_moments = pd.read_csv('../input/champs-scalar-coupling/dipole_moments.csv')

train = pd.merge(train, dipole_moments, how='left',

              left_on='molecule_name',

              right_on='molecule_name')

# calculate the magnitude (attempts at fitting the components using the below properties failed utterly) note that this is in QM9 (mu)

train['dipole_moment'] = train.apply(lambda row: math.sqrt(row['X']**2 + row['Y']**2 + row['Z']**2), axis=1)

train.drop(['X','Y','Z'], axis=1, inplace=True)

del dipole_moments
# potential energy of molecules

potential_energy = pd.read_csv('../input/champs-scalar-coupling/potential_energy.csv')

train = pd.merge(train, potential_energy, how='left',

              left_on='molecule_name',

              right_on='molecule_name')

del potential_energy
# dipole_moment and potential_energy are molecular features,

# so let's get some molecular features and use them to estimate

# count the total no. atoms in the molecule and the number of each type

atom_cnt = structures['molecule_name'].value_counts().reset_index(level=0)

atom_cnt.rename({'index': 'molecule_name', 'molecule_name': 'atom_count'}, axis=1, inplace=True)

train = pd.merge(train, atom_cnt, how='left', on='molecule_name')

test = pd.merge(test, atom_cnt, how='left', on='molecule_name')

del atom_cnt



# (the following should be put in a loop)

H_cnt = structures['molecule_name'][structures['atom']=='H'].value_counts().reset_index(level=0)

H_cnt.rename({'index': 'molecule_name', 'molecule_name': 'H_count'}, axis=1, inplace=True)

train = pd.merge(train, H_cnt, how='left', on='molecule_name')

test = pd.merge(test, H_cnt, how='left', on='molecule_name')

del H_cnt



C_cnt = structures['molecule_name'][structures['atom']=='C'].value_counts().reset_index(level=0)

C_cnt.rename({'index': 'molecule_name', 'molecule_name': 'C_count'}, axis=1, inplace=True)

train = pd.merge(train, C_cnt, how='left', on='molecule_name')

test = pd.merge(test, C_cnt, how='left', on='molecule_name')

del C_cnt



O_cnt = structures['molecule_name'][structures['atom']=='O'].value_counts().reset_index(level=0)

O_cnt.rename({'index': 'molecule_name', 'molecule_name': 'O_count'}, axis=1, inplace=True)

train = pd.merge(train, O_cnt, how='left', on='molecule_name')

test = pd.merge(test, O_cnt, how='left', on='molecule_name')

del O_cnt



N_cnt = structures['molecule_name'][structures['atom']=='N'].value_counts().reset_index(level=0)

N_cnt.rename({'index': 'molecule_name', 'molecule_name': 'N_count'}, axis=1, inplace=True)

train = pd.merge(train, N_cnt, how='left', on='molecule_name')

test = pd.merge(test, N_cnt, how='left', on='molecule_name')

del N_cnt



F_cnt = structures['molecule_name'][structures['atom']=='F'].value_counts().reset_index(level=0)

F_cnt.rename({'index': 'molecule_name', 'molecule_name': 'F_count'}, axis=1, inplace=True)

train = pd.merge(train, F_cnt, how='left', on='molecule_name')

test = pd.merge(test, F_cnt, how='left', on='molecule_name')

del F_cnt



train = train.fillna(0)

test = test.fillna(0)
# get molecular distance props from previous kernel

# https://www.kaggle.com/robertburbidge/distance-features

mol_props = ['molecule_dist_mean', 'molecule_dist_std', 'molecule_dist_skew',

             'molecule_dist_kurt', 'meanx', 'meany', 'meanz', 'meanxH', 'meanyH', 'meanzH',

             'meanxC', 'meanyC', 'meanzC','meanxN', 'meanyN', 'meanzN','meanxO', 'meanyO', 'meanzO',

             'meanxF', 'meanyF', 'meanzF']



train[mol_props] = train_dist[mol_props]

test[mol_props] = test_dist[mol_props]
# electronegativity (this could be better informed by physical chemistry)

atoms = ['H', 'C', 'N', 'O', 'F']

electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

for atom in atoms:

    meanx = (structures[structures['atom']==atom].groupby('molecule_name')['x'].apply(np.mean) *

             electronegativity[atom]).reset_index()

    meanx.rename({'x': 'meanxe' + atom}, axis=1, inplace=True)

    train = pd.merge(train, meanx, how='left', on='molecule_name')

    test = pd.merge(test, meanx, how='left', on='molecule_name')



    meany = (structures[structures['atom']==atom].groupby('molecule_name')['y'].apply(np.mean) *

             electronegativity[atom]).reset_index()

    meany.rename({'y': 'meanye' + atom}, axis=1, inplace=True)

    train = pd.merge(train, meany, how='left', on='molecule_name')

    test = pd.merge(test, meany, how='left', on='molecule_name')



    meanz = (structures[structures['atom']==atom].groupby('molecule_name')['z'].apply(np.mean) *

             electronegativity[atom]).reset_index()

    meanz.rename({'z': 'meanze' + atom}, axis=1, inplace=True)

    train = pd.merge(train, meanz, how='left', on='molecule_name')

    test = pd.merge(test, meanz, how='left', on='molecule_name')

    

del meanx, meany, meanz, structures
# train, predict at molecular level (could have done this earlier but you may want to pull in extra features 

# and aggregate them before this stage, so I left this step to just prior to modelling)

train_mol = train.drop_duplicates(subset=['molecule_name'])

test_mol = test.drop_duplicates(subset=['molecule_name'])
# molecular features for predicting mu

pred_vars = [v for v in train.columns if v not in ['id', 'molecule_name', 'atom_index_0', 'atom_index_1',

                                                   'type', 'dipole_moment', 'potential_energy',

                                                   'scalar_coupling_constant']]
# train-val split by molecule_name (since train and test data have disjoint molecules)

molecule_names = pd.DataFrame(permutation(train_mol['molecule_name'].unique()),columns=['molecule_name'])

nm = molecule_names.shape[0]

ntrn = int(0.9*nm)

nval = int(0.1*nm)



tmp_train = pd.merge(train_mol, molecule_names[0:ntrn], how='right', on='molecule_name')

tmp_val = pd.merge(train_mol, molecule_names[ntrn:nm], how='right', on='molecule_name')



X_train = tmp_train[pred_vars]

X_val = tmp_val[pred_vars]

y_train = tmp_train['dipole_moment']

y_val = tmp_val['dipole_moment']

del tmp_train, tmp_val
# data for LightGBM

train_data = lightgbm.Dataset(X_train, label=y_train)

val_data = lightgbm.Dataset(X_val, label=y_val)
# heuristic parameters for LightGBM

params = { 'objective': 'regression_l1',

           'learning_rate': 0.1,

           'num_leaves': 255,

           'num_threads': -1,

           'bagging_fraction': 0.9,

           'bagging_freq': 10,

           'feature_fraction': 0.9,

           'lambda_l1': 10.0,

           'max_bin': 255,

           'min_child_samples': 50,

           }
# training

model = lightgbm.train(params,

                       train_data,

                       valid_sets=[train_data, val_data], verbose_eval=500,

                       num_boost_round=4000,

                       early_stopping_rounds=100)
# validation

pred_val = model.predict(X_val)

pred_median = np.full(y_val.shape, np.median(y_val))

print(metrics.mean_absolute_error(y_val, pred_val) / metrics.mean_absolute_error(y_val, pred_median))

plt.scatter(y_val, pred_val)
# train & test predictions

pred_train = model.predict(train_mol[pred_vars])

pred_test = model.predict(test_mol[pred_vars])
# save these for inputs for further modelling

train_dipole_moment = pd.DataFrame(train_mol['molecule_name'])

train_dipole_moment['dipole_moment_pred'] = pred_train

train_dipole_moment.to_csv('train_dipole_moment.csv', index=False)

test_dipole_moment = pd.DataFrame(test_mol['molecule_name'])

test_dipole_moment['dipole_moment_pred'] = pred_test

test_dipole_moment.to_csv('test_dipole_moment.csv', index=False)
# molecular features for predicting potential energy

pred_vars = [v for v in train.columns if v not in ['id', 'molecule_name', 'atom_index_0', 'atom_index_1',

                                                   'type', 'dipole_moment', 'potential_energy',

                                                   'scalar_coupling_constant']]
# train-val split by molecule_name

molecule_names = pd.DataFrame(permutation(train_mol['molecule_name'].unique()),columns=['molecule_name'])

nm = molecule_names.shape[0]

ntrn = int(0.9*nm)

nval = int(0.1*nm)



tmp_train = pd.merge(train_mol, molecule_names[0:ntrn], how='right', on='molecule_name')

tmp_val = pd.merge(train_mol, molecule_names[ntrn:nm], how='right', on='molecule_name')



X_train = tmp_train[pred_vars]

X_val = tmp_val[pred_vars]

y_train = tmp_train['potential_energy']

y_val = tmp_val['potential_energy']

del tmp_train, tmp_val
# data for LightGBM

train_data = lightgbm.Dataset(X_train, label=y_train)

val_data = lightgbm.Dataset(X_val, label=y_val)
# heuristic parameters for LightGBM

params = { 'objective': 'regression_l1',

           'learning_rate': 0.1,

           'num_leaves': 255,

           'num_threads': -1,

           'bagging_fraction': 0.9,

           'bagging_freq': 10,

           'feature_fraction': 0.9,

           'lambda_l1': 10.0,

           'max_bin': 255,

           'min_child_samples': 50,

           }
# training

model = lightgbm.train(params,

                       train_data,

                       valid_sets=[train_data, val_data], verbose_eval=100,

                       num_boost_round=1000,

                       early_stopping_rounds=100)
# validation

pred_val = model.predict(X_val)

pred_median = np.full(y_val.shape, np.median(y_val))

print(metrics.mean_absolute_error(y_val, pred_val) / metrics.mean_absolute_error(y_val, pred_median))

plt.scatter(y_val, pred_val)
# train & test predictions

pred_train = model.predict(train_mol[pred_vars])

pred_test = model.predict(test_mol[pred_vars])
# save these for inputs for further modelling

train_potential_energy = pd.DataFrame(train_mol['molecule_name'])

train_potential_energy['potential_energy_pred'] = pred_train

train_potential_energy.to_csv('train_potential_energy.csv', index=False)

test_potential_energy = pd.DataFrame(test_mol['molecule_name'])

test_potential_energy['potential_energy_pred'] = pred_test

test_potential_energy.to_csv('test_potential_energy.csv', index=False)
# example of using these features to improve (slightly) on previous performance with dist features

train.drop(['potential_energy', 'dipole_moment'], axis=1, inplace=True)

train = pd.merge(train, train_dipole_moment, how='left', on='molecule_name')

train = pd.merge(train, train_potential_energy, how='left', on='molecule_name')

test = pd.merge(test, test_dipole_moment, how='left', on='molecule_name')

test = pd.merge(test, test_potential_energy, how='left', on='molecule_name')

del train_dipole_moment, train_potential_energy, test_dipole_moment, test_potential_energy
# get remaining features from previous kernel

addfeats = [v for v in train_dist.columns if v not in mol_props]

train[addfeats] = train_dist[addfeats]

test[addfeats] = test_dist[addfeats]
# https://www.kaggle.com/artgor/artgor-utils

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024 ** 2

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

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (

                start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
# features for prediction

pred_vars = [v for v in train.columns if v not in ['id', 'molecule_name', 'scalar_coupling_constant']]
# train-val split by molecule_name

molecule_names = pd.DataFrame(permutation(train['molecule_name'].unique()),columns=['molecule_name'])

nm = molecule_names.shape[0]

ntrn = int(0.9*nm)

nval = int(0.1*nm)



tmp_train = pd.merge(train, molecule_names[0:ntrn], how='right', on='molecule_name')

tmp_val = pd.merge(train, molecule_names[ntrn:nm], how='right', on='molecule_name')



X_train = tmp_train[pred_vars]

X_val = tmp_val[pred_vars]

y_train = tmp_train['scalar_coupling_constant']

y_val = tmp_val['scalar_coupling_constant']

del tmp_train, tmp_val
# heuristic parameters for LightGBM

params = { 'objective': 'regression_l1',

           'learning_rate': 0.1,

           'num_leaves': 1023,

           'num_threads': -1,

           'bagging_fraction': 0.5,

           'bagging_freq': 1,

           'feature_fraction': 0.9,

           'lambda_l1': 10.0,

           'max_bin': 255,

           'min_child_samples': 15,

           }
# categorical features (note that these are already integer-coded)

cat_feats = ['type', 'type_0', 'type_1', 'atom_0l', 'atom_0r', 'atom_1l', 'atom_1r']
# data for lightgbm

train_data = lightgbm.Dataset(X_train, label=y_train, categorical_feature=cat_feats)

val_data = lightgbm.Dataset(X_val, label=y_val, categorical_feature=cat_feats)
# training

model = lightgbm.train(params,

                       train_data,

                       valid_sets=[train_data, val_data], verbose_eval=500,

                       num_boost_round=4000,

                       early_stopping_rounds=100)
# evaluation metric for validation

# https://www.kaggle.com/abhishek/competition-metric

def metric(df, preds):

    df["prediction"] = preds

    maes = []

    for t in df.type.unique():

        y_true = df[df.type==t].scalar_coupling_constant.values # column 1 is the target whatever the y_var

        y_pred = df[df.type==t].prediction.values

        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

        maes.append(mae)

    return np.mean(maes)
# validation metric

preds = model.predict(X_val)

metric(pd.concat([X_val, y_val], axis=1), preds)
# submission

preds_sub = model.predict(test[pred_vars])

sub['scalar_coupling_constant'] = preds_sub

sub.to_csv('submission_feats_dist_mol01.csv', index=False)