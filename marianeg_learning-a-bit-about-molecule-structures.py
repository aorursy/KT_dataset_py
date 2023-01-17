# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from numpy.random import permutation

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import lightgbm as lgb



from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

os.listdir("../input")



# Any results you write to the current directory are saved as output.
train = pd.read_csv(f'../input/train.csv')

test = pd.read_csv(f'../input/test.csv')

structures = pd.read_csv(f'../input/structures.csv')



train[train['molecule_name'] == "dsgdb9nsd_000001"]
# Adapted from the Andrew Lukyanenko's kernel. 

# https://www.kaggle.com/artgor/molecular-properties-eda-and-models

def map_atom_info(df, df2, atom_idx, col_names):

    df = pd.merge(df, df2, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    for i in range(len(col_names)):

        df = df.rename(columns={f'{col_names[i]}': f'{col_names[i]}_{atom_idx}'})

    return df
cols_change = ['x','y','z','atom']

train = map_atom_info(train, structures, 0, cols_change)

train = map_atom_info(train, structures, 1, cols_change)



test = map_atom_info(test, structures, 0, cols_change)

test = map_atom_info(test, structures, 1, cols_change)



list(test.columns)
train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

train['dist_x'] = (train['x_0'] - train['x_1']) ** 2

test['dist_x'] = (test['x_0'] - test['x_1']) ** 2

train['dist_y'] = (train['y_0'] - train['y_1']) ** 2

test['dist_y'] = (test['y_0'] - test['y_1']) ** 2

train['dist_z'] = (train['z_0'] - train['z_1']) ** 2

test['dist_z'] = (test['z_0'] - test['z_1']) ** 2
# LabelEncoding the the character values, splitting and reducing the dataframe sizes

for f in ['atom_0', 'atom_1', 'type']:

    lbl = LabelEncoder()

    lbl.fit(list(train[f].values) + list(test[f].values))

    lbl_name_mapping = dict(zip(lbl.classes_, lbl.transform(lbl.classes_)))

    train[f] = lbl.transform(list(train[f].values))

    test[f] = lbl.transform(list(test[f].values))

    print(lbl_name_mapping)
train.groupby('type')['scalar_coupling_constant'].agg(

    ['count', 'std', 'mean', 'median', 'var', 'skew']).reset_index()
# This memory reduction function will make the execution faster

# https://www.kaggle.com/artgor/artgor-utils

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

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float64).precision:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
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
# Splitting the "train set" in two to allow internal model validation (adapted from)

# https://www.kaggle.com/robertburbidge/using-estimated-mulliken-charges

def split_data(df):

    

    molecule_names = pd.DataFrame(permutation(df['molecule_name'].unique()),columns=['molecule_name'])

    nm = molecule_names.shape[0]

    ntrn = int(0.9*nm)

    nval = int(0.1*nm)



    tmp_train = pd.merge(df, molecule_names[0:ntrn], how='right', on='molecule_name')

    tmp_train = reduce_mem_usage(tmp_train)

    tmp_val = pd.merge(df, molecule_names[ntrn:nm], how='right', on='molecule_name')

    tmp_val = reduce_mem_usage(tmp_val)



    X_train = tmp_train.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)

    y_train = tmp_train['scalar_coupling_constant']



    X_val = tmp_val.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)

    y_val = tmp_val['scalar_coupling_constant']



    return X_train, y_train, X_val, y_val
X_train, y_train, X_val, y_val = split_data(train)
params = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'regression_l1',

          'max_depth': 13,

          'learning_rate': 0.2,

          'subsample_freq': 1,

          'subsample': 0.9,

          'bagging_seed': 11,

          'metric': 'mae',

          'verbosity': -1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.3,

          'colsample_bytree'

          : 1.0

         }



model_basic = lgb.LGBMRegressor(**params, n_estimators = 5000, n_jobs = -1)

model_basic.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_val, y_val)],

                    verbose=500, early_stopping_rounds=100)
pred_train = model_basic.predict(X_train)

pred_train_median = np.full(y_train.shape, np.median(y_train))

print(metrics.mean_absolute_error(y_train, pred_train) / metrics.mean_absolute_error(y_train, pred_train_median))

metric(pd.concat([X_train, y_train], axis=1), pred_train)
pred_train_df = pd.DataFrame(pred_train, columns=['pred_sc'])

X_train_types = pd.concat([X_train.reset_index(drop=True), pred_train_df], axis=1)

X_train_types.groupby('type')['pred_sc'].agg(

    ['count', 'std', 'mean', 'median', 'var', 'skew']).reset_index()
X_train_types = pd.concat([X_train_types.reset_index(drop=True), y_train.rename('original_sc')], axis=1)

sns.lmplot(data=X_train_types, x="pred_sc", y="original_sc", hue="type");
pred_val = model_basic.predict(X_val)

pred_val_median = np.full(y_val.shape, np.median(y_val))

print(metrics.mean_absolute_error(y_val, pred_val) / metrics.mean_absolute_error(y_val, pred_val_median))

metric(pd.concat([X_val, y_val], axis=1), pred_val)



pred_val_df = pd.DataFrame(pred_val, columns=['pred_sc'])

X_val_types = pd.concat([X_val.reset_index(drop=True), pred_val_df], axis=1)

X_val_types = pd.concat([X_val_types.reset_index(drop=True), y_val.rename('original_sc')], axis=1)

sns.lmplot(data=X_val_types, x="pred_sc", y="original_sc", hue="type");
# We can't forget the memory usage reduction

test = reduce_mem_usage(test)

X_test = test.drop(['id', 'molecule_name'], axis=1)



pred_test = model_basic.predict(X_test)

sub = pd.read_csv(f'../input/sample_submission.csv')

sub['scalar_coupling_constant'] = pred_test

sub.head()
sub.to_csv('submission.csv', index=False)