import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold, RandomizedSearchCV

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.utils import resample

import xgboost as xgb

import matplotlib.pyplot as plt

from IPython.display import clear_output

import time, datetime

%matplotlib inline
PATH_TRAIN = '../input/train.csv'

PATH_TEST =  '../input/test.csv'

PATH_STR = '../input/structures.csv'



train = pd.read_csv(PATH_TRAIN)

LEN_TRAIN = len(train)

test = pd.read_csv(PATH_TEST)

LEN_TEST = len(test)
# map structure information into train/test set.

structures = pd.read_csv(PATH_STR)

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



start = time.time()



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)



print(str(datetime.timedelta(seconds=time.time()-start)))

print(train.columns)
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

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
# get new distance columns

start = time.time()

train_c0 = train[['x_0', 'y_0', 'z_0']].values

train_c1 = train[['x_1', 'y_1', 'z_1']].values

train['dist'] = np.linalg.norm(train_c0 - train_c1, axis=1)

test_c0 = test[['x_0', 'y_0', 'z_0']].values

test_c1 = test[['x_1', 'y_1', 'z_1']].values

test['dist'] = np.linalg.norm(test_c0 - test_c1, axis=1)



train['x_dist'] = (train['x_0'] - train['x_1']) ** 2

train['y_dist'] = (train['y_0'] - train['y_1']) ** 2

train['z_dist'] = (train['z_0'] - train['z_1']) ** 2

test['x_dist'] = (test['x_0'] - test['x_1']) ** 2

test['y_dist'] = (test['y_0'] - test['y_1']) ** 2

test['z_dist'] = (test['z_0'] - test['z_1']) ** 2



train['xy_dist'] = np.linalg.norm(train[['x_0', 'y_0']].values - train[['x_1', 'y_1']].values, axis=1)

train['xz_dist'] = np.linalg.norm(train[['x_0', 'z_0']].values - train[['x_1', 'z_1']].values, axis=1)

train['yz_dist'] = np.linalg.norm(train[['y_0', 'z_0']].values - train[['y_1', 'z_1']].values, axis=1)

test['xy_dist'] = np.linalg.norm(test[['x_0', 'y_0']].values - test[['x_1', 'y_1']].values, axis=1)

test['xz_dist'] = np.linalg.norm(test[['x_0', 'z_0']].values - test[['x_1', 'z_1']].values, axis=1)

test['yz_dist'] = np.linalg.norm(test[['y_0', 'z_0']].values - test[['y_1', 'z_1']].values, axis=1)

print(str(datetime.timedelta(seconds=time.time()-start)))

print(train.columns)
fig = plt.figure(figsize=(15, 5))

fig.suptitle('Counts of types')

ax1 = fig.add_subplot(121)

ax1.set_ylim(0, 1600000)

ax1.bar(train['type'].value_counts().index, train['type'].value_counts(sort=False).values, color='k', width=0.5)

ax1.title.set_text('Train set')



ax2 = fig.add_subplot(122)

ax2.set_ylim(0, 1600000)

ax2.bar(test['type'].value_counts().index, test['type'].value_counts(sort=False).values, color='r', width=0.5)

ax2.title.set_text('Test set')

plt.show()
fig = plt.figure(figsize=(15, 5))

fig.suptitle('Counts of atom1')



ax1 = fig.add_subplot(121)

ax1.set_ylim(0, 4000000)

ax1.bar(train['atom_1'].value_counts().index, train['atom_1'].value_counts(sort=False).values, color='k', width=0.5)

ax1.title.set_text('Train set_atom1')



ax2 = fig.add_subplot(122)

ax2.set_ylim(0, 4000000)

ax2.bar(test['atom_1'].value_counts().index, test['atom_1'].value_counts(sort=False).values, color='r', width=0.5)

ax2.title.set_text('Test set_atom1')

plt.show()
train.hist(column='scalar_coupling_constant', bins=20, color='k', grid=False)
train.hist(column='scalar_coupling_constant', by='type', figsize=(10, 10), bins=20, color='k')
train.hist(column='scalar_coupling_constant', by='atom_1', figsize=(7, 7), bins=20, color='k')
resample(train, n_samples=10000).plot(kind='scatter', x='dist', y='scalar_coupling_constant', color='k', s=10, alpha=0.3)
rand_idx = np.random.randint(0, LEN_TRAIN, 1000)

fig = plt.figure(figsize=(15, 5))

fig.suptitle('Cordinate values and scalar coupling')

ax1 = fig.add_subplot(121)

train.iloc[rand_idx].plot(kind='scatter', x='x_0', y='scalar_coupling_constant', color='r', s=5, alpha=0.3, label='x_0', ax=ax1)

train.iloc[rand_idx].plot(kind='scatter', x='y_0', y='scalar_coupling_constant', color='g', s=5, alpha=0.3, label='y_0', ax=ax1)

train.iloc[rand_idx].plot(kind='scatter', x='z_0', y='scalar_coupling_constant', color='b', s=5, alpha=0.3, label='z_0', ax=ax1)

ax1.set_xlabel('cordinate 0')

ax2 = fig.add_subplot(122)

train.iloc[rand_idx].plot(kind='scatter', x='x_1', y='scalar_coupling_constant', color='r', s=5, alpha=0.3, label='x_1', ax=ax2)

train.iloc[rand_idx].plot(kind='scatter', x='y_1', y='scalar_coupling_constant', color='g', s=5, alpha=0.3, label='y_1', ax=ax2)

train.iloc[rand_idx].plot(kind='scatter', x='z_1', y='scalar_coupling_constant', color='b', s=5, alpha=0.3, label='z_1', ax=ax2)

ax2.set_xlabel('cordinate 1')

plt.show()
rand_idx = np.random.randint(0, LEN_TRAIN, 1000)

fig = plt.figure(figsize=(15, 5))

fig.suptitle('1-d, 2-d disntaces and scalar coupling')

ax1 = fig.add_subplot(121)

train.iloc[rand_idx].plot(kind='scatter', x='x_dist', y='scalar_coupling_constant', color='r', s=5, alpha=0.3, label='x_dist', ax=ax1)

train.iloc[rand_idx].plot(kind='scatter', x='y_dist', y='scalar_coupling_constant', color='g', s=5, alpha=0.3, label='y_dist', ax=ax1)

train.iloc[rand_idx].plot(kind='scatter', x='z_dist', y='scalar_coupling_constant', color='b', s=5, alpha=0.3, label='z_dist', ax=ax1)

ax1.set_xlabel('1d_dist')

ax2 = fig.add_subplot(122)

train.iloc[rand_idx].plot(kind='scatter', x='xy_dist', y='scalar_coupling_constant', color='r', s=5, alpha=0.3, label='xy_dist', ax=ax2)

train.iloc[rand_idx].plot(kind='scatter', x='xz_dist', y='scalar_coupling_constant', color='g', s=5, alpha=0.3, label='xz_dist', ax=ax2)

train.iloc[rand_idx].plot(kind='scatter', x='yz_dist', y='scalar_coupling_constant', color='b', s=5, alpha=0.3, label='yz_dist', ax=ax2)

ax2.set_xlabel('2d_dist')

plt.show()
train['type_0'] = train['type'].apply(lambda x: int(x[0]))

test['type_0'] = test['type'].apply(lambda x: int(x[0]))

def create_more_features(df):

    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')

    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')

    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')

    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')

    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')

    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')

    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']

    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')

    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')

    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')

    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']

    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']

    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')

    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']

    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']

    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')

    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']

    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']

    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')

    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']

    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']

    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')

    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']

    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']

    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')

    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']

    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']

    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')

    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']

    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']

    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')

    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']

    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']

    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')

    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')

    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']

    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']

    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')

    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']

    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')

    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']

    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')

    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']

    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']

    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')

    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')

    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')

    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']



    return df



train = reduce_mem_usage(create_more_features(train)).fillna(0)

test = reduce_mem_usage(create_more_features(test)).fillna(0)
# transform categorical columns

def transform_onehot(df, col_name):

    enc = OneHotEncoder()

    arr = enc.fit_transform(df[col_name].values.reshape(-1, 1)).toarray().astype(int)

    col_names = [col_name+'_'+cat for cat in enc.categories_[0]]

    new_col = pd.DataFrame(arr, columns=col_names)

    ret = df.join(new_col)

    return ret.drop(columns=col_name)



train = transform_onehot(train, 'atom_0')

train = transform_onehot(train, 'atom_1')



test = transform_onehot(test, 'atom_0')

test = transform_onehot(test, 'atom_1')



train.head()
# preprocessing is ended, then start the regression

##############

# i don't need reference columns anymore

train.drop(columns=['id', 'molecule_name', 'atom_0_H'], inplace=True)

test.drop(columns=['molecule_name', 'atom_0_H'], inplace=True)
# Finding important features in order

type_model = dict()

for type_ in ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']:

    print(f'KFold training started for {type_} model.')

    X_train = train.loc[train['type'] == type_].drop(columns=['type', 'scalar_coupling_constant'])

    y_train = train.loc[train['type'] == type_, 'scalar_coupling_constant']

    folds = KFold(n_splits=5, shuffle=True)

    X_train_sub = resample(X_train, n_samples=int(len(X_train)/1000), random_state=13)

    y_train_sub = resample(y_train, n_samples=int(len(X_train)/1000), random_state=13)

    importances = pd.Series(index=X_train.columns).fillna(0)

    for i, (idx_train, idx_test) in enumerate(folds.split(X_train_sub.values, y_train_sub.values)):

        print(f'{i+1} / 5 folds')

        X_train_folds = X_train_sub.iloc[idx_train]

        y_train_folds = y_train_sub.iloc[idx_train]

        X_valid_folds = X_train_sub.iloc[idx_train]

        y_valid_folds = y_train_sub.iloc[idx_train]

        

        # using 3 regression algoritms

        #abr = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, loss='exponential')

        #gbr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=10, criterion='mae')

        xgb_ = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=10)

        

        #abr.fit(X_train_folds, y_train_folds)

        #gbr.fit(X_train_folds, y_train_folds)

        xgb_.fit(X_train_folds, y_train_folds)

        

        #importances += abr.feature_importances_

        #importances += gbr.feature_importances_

        importances += xgb_.feature_importances_

        

    type_model[type_] = importances
type_model['3JHN'].sort_values(ascending=False)[:15].sort_values().plot(kind='barh')
type_model['2JHC'].sort_values(ascending=False)[:15].sort_values().plot(kind='barh')
important_features = dict()

for model, values in type_model.items():

    important_features[model] = values.sort_values(ascending=False)[:15].index



important_features
#Random grid searching with XGBRegressor

xgb_b = dict()

xgb_score = 0

for type_ in ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']:

    print(f'XGBoost Regressor : Random searching stated for {type_} model.')

    X_train = train.loc[train['type'] == type_, important_features[type_]]

    y_train = train.loc[train['type'] == type_, 'scalar_coupling_constant']

    X_train_sub = resample(X_train, n_samples=int(len(X_train)/1000), random_state=43)

    y_train_sub = resample(y_train, n_samples=int(len(X_train)/1000), random_state=43)

    params={'learning_rate': [0.01, 0.1, 1.0],

                    'min_child_weight': [1, 5, 10],

                    'n_estimators': [50, 100, 200, 400, 800],

                    'max_depth': [4, 10, 15, 20],

                    'gamma': [0.5, 1, 1.5, 2, 5],

                    'subsample': [0.1, 0.5, 1.0],

                    'colsample_bytree': [0.1, 0.5, 1.0],

                    'colsample_bylevel': [0.1, 0.5, 1.0],

                    'scale_pos_weight': [0.01, 0.1, 1.0],

                    'reg_lambda': [0.1, 1, 10],

                    'reg_alpha': [0.01, 0.1, 1.0],

                    'max_delta_step': [0, 1, 10],

                    'scale_pos_weight': [0.01, 0.1, 1]}

    cv_xgb = RandomizedSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'),

                                   param_distributions=params,

                                   cv=KFold(n_splits=5, shuffle=True),

                                scoring='neg_mean_absolute_error',

                                   n_iter=15,

                                   verbose=1)

    cv_xgb.fit(X_train_sub.values, y_train_sub.values)

    xgb_b[type_] = cv_xgb.best_estimator_

    xgb_score += cv_xgb.best_score_

    

"""

#Grid searching with Adaboost regressor

abr_b = dict()

abr_score = 0

for type_ in ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']:

    print(f'Adaboost Regressor : Random searching started for {type_} model.')

    X_train = train.loc[train['type'] == type_, important_features[type_]]

    y_train = train.loc[train['type'] == type_, 'scalar_coupling_constant']

    X_train_sub = resample(X_train, n_samples=int(len(X_train)/1000), random_state=23)

    y_train_sub = resample(y_train, n_samples=int(len(X_train)/1000), random_state=23)

    params={'n_estimators': [50, 100, 200, 400, 800],

            'learning_rate':[0.01, 0.1, 1],

            'loss': ['linear', 'square', 'exponential']

            }

    cv_abr = RandomizedSearchCV(estimator=AdaBoostRegressor(),

                                   param_distributions=params,

                                   cv=KFold(n_splits=5, shuffle=True),

                                scoring='neg_mean_absolute_error',

                                   n_iter=15,

                                   verbose=1)

    cv_abr.fit(X_train_sub.values, y_train_sub.values)

    abr_b[type_] = cv_abr.best_estimator_

    abr_score += cv_abr.best_score_





#Random grid searching with GradientBoosting regressor

gbr_b = dict()

gbr_score = 0

for type_ in ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']:

    print(f'Gradient Boosting Regressor : Random searching stated for {type_} model.')

    X_train = train.loc[train['type'] == type_, important_features[type_]]

    y_train = train.loc[train['type'] == type_, 'scalar_coupling_constant']

    X_train_sub = resample(X_train, n_samples=int(len(X_train)/1000), random_state=33)

    y_train_sub = resample(y_train, n_samples=int(len(X_train)/1000), random_state=33)

    params={'n_estimators': [50, 100, 200, 400, 800],

            'learning_rate':[0.01, 0.1, 1],

            'loss': ['ls', 'lad', 'huber', 'quantile'],

            'subsample' : [0.01, 0.1, 1],

            'min_samples_split': [0.01, 0.1, 2],

            'min_samples_leaf': [0.01, 0.1, 2],

            'max_depth': [4, 10, 15, 20],

            'max_features':['auto', 'sqrt', 'log2']

            }

    cv_gbr = RandomizedSearchCV(estimator=GradientBoostingRegressor(criterion='mae'),

                                   param_distributions=params,

                                   cv=KFold(n_splits=5, shuffle=True),

                                scoring='neg_mean_absolute_error',

                                   n_iter=15,

                                   verbose=1)

    cv_gbr.fit(X_train_sub.values, y_train_sub.values)

    gbr_b[type_] = cv_gbr.best_estimator_

    gbr_score += cv_gbr.best_score_

"""
"""

# Training and prediction with XGB regressor

xgb_predicted = pd.Series(index=test.index)

for type_ in ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']:

    print(f'{type_} type training started.')

    X_train = train.loc[train['type'] == type_, important_features[type_]]

    y_train = train.loc[train['type'] == type_, 'scalar_coupling_constant']

    X_test = test.loc[test['type'] == type_, important_features[type_]]

    clf_xgb = xgb_b[type_]

    clf_xgb.fit(X_train.values, y_train.values)

    result = pd.Series(data=clf_xgb.predict(X_test.values), index=X_test.index)

    xgb_predicted = xgb_predicted.add(result, fill_value=0)



# Training and prediction with AdaBoost regressor

abr_predicted = pd.Series(index=test.index)

for type_ in ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']:

    print(f'{type_} type training started.')

    X_train = train.loc[train['type'] == type_, important_features[type_]]

    y_train = train.loc[train['type'] == type_, 'scalar_coupling_constant']

    X_test = test.loc[test['type'] == type_, important_features[type_]]

    clf_abr = abr_b[type_]

    clf_abr.fit(X_train.values, y_train.values)

    result = pd.Series(data=clf_abr.predict(X_test.values), index=X_test.index)

    abr_predicted = abr_predicted.add(result, fill_value=0)



# Training and prediction with GradientBoosting regressor

gbr_predicted = pd.Series(index=test.index)

for type_ in ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']:

    print(f'{type_} type training started.')

    X_train = train.loc[train['type'] == type_, important_features[type_]]

    y_train = train.loc[train['type'] == type_, 'scalar_coupling_constant']

    X_test = test.loc[test['type'] == type_, important_features[type_]]

    clf_gbr = gbr_b[type_]

    clf_gbr.fit(X_train.values, y_train.values)

    result = pd.Series(data=clf_gbr.predict(X_test.values), index=X_test.index)

    gbr_predicted = gbr_predicted.add(result, fill_value=0)

"""
"""

# average the predicted target values

#predicted = (xgb_predicted + abr_predicted + gbr_predicted) / 3



# make a submission file

#pd.DataFrame({'id':test['id'], 'scalar_coupling_constant':predicted}).to_csv('predicted.csv', index=False)

#pd.DataFrame({'id':test['id'], 'scalar_coupling_constant':abr_predicted}).to_csv('abr_predicted.csv', index=False)

#pd.DataFrame({'id':test['id'], 'scalar_coupling_constant':gbr_predicted}).to_csv('gbr_predicted.csv', index=False)

pd.DataFrame({'id':test['id'], 'scalar_coupling_constant':xgb_predicted}).to_csv('xgb_predicted.csv', index=False)

"""