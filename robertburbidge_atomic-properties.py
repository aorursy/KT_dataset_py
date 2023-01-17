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

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
# get xyz data for each atom

structures = pd.read_csv('../input/structures.csv')
# atomic properties

# https://www.lenntech.com/periodic-chart-elements/

atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71, np.nan: 0}

atomic_number = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, np.nan: 0}

atomic_mass = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984, np.nan: 0}

vanderwaalsradius = {'H': 120, 'C': 185, 'N': 154, 'O': 140, 'F': 135, np.nan: 0}

covalenzradius = {'H': 30, 'C': 77, 'N': 70, 'O': 66, 'F': 58, np.nan: 0}

electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, np.nan: 0}

ionization_energy = {'H': 13.5984, 'C': 11.2603, 'N': 14.5341, 'O': 13.6181, 'F': 17.4228, np.nan: np.inf}
def atom_props(df, suffix):

    df['atomic_radius' + suffix] = df['atom_' + suffix].apply(lambda x: atomic_radius[x])

    df['atomic_protons' + suffix] = df['atom_' + suffix].apply(lambda x: atomic_number[x])

    df['atomic_mass' + suffix] = df['atom_' + suffix].apply(lambda x: atomic_mass[x])

    df['vanderwaalsradius' + suffix] = df['atom_' + suffix].apply(lambda x: vanderwaalsradius[x])

    df['covalenzradius' + suffix] = df['atom_' + suffix].apply(lambda x: covalenzradius[x])

    df['electronegativity' + suffix] = df['atom_' + suffix].apply(lambda x: electronegativity[x])

    df['ionization_energy' + suffix] = df['atom_' + suffix].apply(lambda x: ionization_energy[x])

    return df
# atom_0, atom_1

train = pd.merge(train, structures[['molecule_name', 'atom_index', 'atom']], how='left',

                 left_on=['molecule_name', 'atom_index_0'],

                 right_on=['molecule_name', 'atom_index']).rename({'atom': 'atom_0'}, axis=1)

train = pd.merge(train, structures[['molecule_name', 'atom_index', 'atom']], how='left',

                 left_on=['molecule_name', 'atom_index_1'],

                 right_on=['molecule_name', 'atom_index']).rename({'atom': 'atom_1'}, axis=1)

test = pd.merge(test, structures[['molecule_name', 'atom_index', 'atom']], how='left',

                 left_on=['molecule_name', 'atom_index_0'],

                 right_on=['molecule_name', 'atom_index']).rename({'atom': 'atom_0'}, axis=1)

test = pd.merge(test, structures[['molecule_name', 'atom_index', 'atom']], how='left',

                 left_on=['molecule_name', 'atom_index_1'],

                 right_on=['molecule_name', 'atom_index']).rename({'atom': 'atom_1'}, axis=1)
# no. atoms in molecule

atom_cnt = structures['molecule_name'].value_counts().reset_index(level=0)

atom_cnt.rename({'index': 'molecule_name', 'molecule_name': 'atom_count'}, axis=1, inplace=True)

train = pd.merge(train, atom_cnt, how='left', on='molecule_name')

test = pd.merge(test, atom_cnt, how='left', on='molecule_name')

del atom_cnt
# neighbours by index (replace the integer codes with the original atom chars)

def lr(df):

    df['atom_index_0l'] = df['atom_index_0'].apply(lambda i: max(i - 1, 0))

    tmp = df[['atom_index_0', 'atom_count']]

    df['atom_index_0r'] = tmp.apply(lambda row: min(row['atom_index_0'] + 1, row['atom_count']), axis=1)

    df = pd.merge(df, structures[['molecule_name', 'atom_index', 'atom']], how='left',

                     left_on=['molecule_name', 'atom_index_0l'],

                     right_on=['molecule_name', 'atom_index']).rename({'atom': 'atom_0l'}, axis=1)

    df = pd.merge(df, structures[['molecule_name', 'atom_index', 'atom']], how='left',

                     left_on=['molecule_name', 'atom_index_0r'],

                     right_on=['molecule_name', 'atom_index']).rename({'atom': 'atom_0r'}, axis=1)

    df['atom_index_1l'] = df['atom_index_1'].apply(lambda i: max(i - 1, 0))

    tmp = df[['atom_index_1', 'atom_count']]

    df['atom_index_1r'] = tmp.apply(lambda row: min(row['atom_index_1'] + 1, row['atom_count']), axis=1)

    df = pd.merge(df, structures[['molecule_name', 'atom_index', 'atom']], how='left',

                     left_on=['molecule_name', 'atom_index_1l'],

                     right_on=['molecule_name', 'atom_index']).rename({'atom': 'atom_1l'}, axis=1)

    df = pd.merge(df, structures[['molecule_name', 'atom_index', 'atom']], how='left',

                     left_on=['molecule_name', 'atom_index_1r'],

                     right_on=['molecule_name', 'atom_index']).rename({'atom': 'atom_1r'}, axis=1)

    return df
train = lr(train)

test = lr(test)
# get atomic properties of both atoms and their neighbours

train = atom_props(train, '0')

train = atom_props(train, '0l')

train = atom_props(train, '0r')

train = atom_props(train, '1')

train = atom_props(train, '1l')

train = atom_props(train, '1r')

test = atom_props(test, '0')

test = atom_props(test, '0l')

test = atom_props(test, '0r')

test = atom_props(test, '1')

test = atom_props(test, '1l')

test = atom_props(test, '1r')
# drop duplicate columns

train.drop(['atom_index_x', 'atom_index_y'], axis=1, inplace=True)

test.drop(['atom_index_x', 'atom_index_y'], axis=1, inplace=True)
# https://www.kaggle.com/c/champs-scalar-coupling/discussion/96655#latest-558745

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

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

                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
atom_feats = [v for v in train.columns if v not in ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type',

       'scalar_coupling_constant', 'atom_0', 'atom_1', 'atom_index_0l',

       'atom_count', 'atom_index_0r', 'atom_0l', 'atom_0r', 'atom_1l', 'atom_1r', 'atom_index_1l', 'atom_index_1r']]
# save these for future use

train[['id'] + atom_feats].to_csv('train_atom_feats.csv', index=False)

test[['id'] + atom_feats].to_csv('test_atom_feats.csv', index=False)
# encode label

lbl = LabelEncoder()

lbl.fit(list(train['type'].values) + list(test['type'].values))

train['type'] = lbl.transform(list(train['type'].values))

test['type'] = lbl.transform(list(test['type'].values))
# features for prediction

pred_vars = ['type'] + atom_feats
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
# categorical features for lightgbm

cat_feats = ['type']
# datasets for lightgbm

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

        y_true = df[df.type==t].scalar_coupling_constant.values

        y_pred = df[df.type==t].prediction.values

        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

        maes.append(mae)

    return np.mean(maes)
# validation

preds = model.predict(X_val)

metric(pd.concat([X_val, y_val], axis=1), preds)
# submission

preds_sub = model.predict(test[pred_vars])

sub['scalar_coupling_constant'] = preds_sub

sub.to_csv('submission_atom_feats.csv', index=False)