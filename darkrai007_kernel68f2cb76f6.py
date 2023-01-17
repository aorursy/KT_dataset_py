# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
print(train['molecule_name'].nunique())

print(train['scalar_coupling_constant'].nunique())

print(train['atom_index_0'].nunique())

print(train['atom_index_1'].nunique())

print(train['type'].nunique())
#pd.to_numeric(train['scalar_coupling_constant'],downcast='integer')

#train['scalar_coupling_constant'].astype(int).nunique()
# train['scalar_coupling_constant']=train['scalar_coupling_constant'].astype(int)

# train.head(5)
structure = pd.read_csv("../input/structures.csv")

print(structure.shape)

structure.head()
%matplotlib inline

import matplotlib.pyplot as plt

#train_sample=train.values[::100]

#plt.plot(train_sample.atom_index_0,train_sample.scalar_coupling_constant)

#train_sample[0].shape
# from numpy import array

# train_sample=array([[int(x[0]),x[1],int(x[2]),int(x[3]),x[4],int(x[5])] for x in train_sample])

# #plt.plot(train_sample[2],train_sample[5])

# type(train_sample)
# plt.plot(train_sample[:,2],train_sample[:,5])
# from sklearn.tree import DecisionTreeRegressor

# from sklearn.model_selection import cross_val_score

# reg=DecisionTreeRegressor()

# cross_val_score(reg,train_sample[:,2:4],train_sample[:,5])

import lightgbm as lgb
#  params = {'boosting': 'gbdt', 'colsample_bytree': 1, 

#               'learning_rate': 0.1, 'max_depth': 40, 'metric': 'mae',

#               'min_child_samples': 50, 'num_leaves': 500, 

#               'objective': 'regression', 'reg_alpha': 0.5, 

#               'reg_lambda': 0.8, 'subsample': 0.5}


#     X_train = X_train.drop(['id', 'atom_0', 'type', 'atom_1','molecule_name'], axis=1).values

#     y_train = y_train.values

#     X_val = X_val.drop(['id', 'atom_0', 'type', 'atom_1','molecule_name'], axis=1).values

#     y_val = y_val.values

# train['type'] = train['type'].astype('category')

# train['atom_index_0'] = train['atom_index_0'].astype('category')

# train['atom_index_0'] = train['atom_index_0'].astype('category')

# lgtrain = lgb.Dataset(train.drop(['id','molecule_name'], axis=1))



#lgtrain.set_categorical_feature(categorical_feature=4)

#     lgval = lightgbm.Dataset(X_val, label=y_val)

 

#model_lgb = lgb.train(params, lgtrain, 5000, verbose_eval=500, categorical_feature=['atom_index_0', 'type', 'atom_index_1'])
test = pd.read_csv("../input/test.csv")

#test=test.drop(['id', 'molecule_name'], axis=1)

#test['type'] = test['type'].astype('category')

#test['atom_index_0'] = test['atom_index_0'].astype('category')

#test['atom_index_0'] = test['atom_index_0'].astype('category')
def map_atom_info(df, atom_idx):

    df = pd.merge(df, structure, how = 'left',

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

train.head()
test.head()
str(train['id'][1]).isnumeric()
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
train.head()
train['type_0'] = train['type'].apply(lambda x: x[0])

test['type_0'] = test['type'].apply(lambda x: x[0])

train['type_1'] = train['type'].apply(lambda x: x[1:])

test['type_1'] = test['type'].apply(lambda x: x[1:])

train.head()
train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')

test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')



train['dist_to_type_0_mean'] = train['dist'] / train.groupby('type_0')['dist'].transform('mean')

test['dist_to_type_0_mean'] = test['dist'] / test.groupby('type_0')['dist'].transform('mean')



train['dist_to_type_1_mean'] = train['dist'] / train.groupby('type_1')['dist'].transform('mean')

test['dist_to_type_1_mean'] = test['dist'] / test.groupby('type_1')['dist'].transform('mean')



train[f'molecule_type_dist_mean'] = train.groupby(['molecule_name', 'type'])['dist'].transform('mean')

test[f'molecule_type_dist_mean'] = test.groupby(['molecule_name', 'type'])['dist'].transform('mean')



train.head()
test.head()
import networkx as nx

fig, ax = plt.subplots(figsize = (20, 12))

for i, t in enumerate(train['type'].unique()):

    train_type = train.loc[train['type'] == t]

    

    bad_atoms_0 = list(train_type['atom_index_0'].value_counts(normalize=True)[train_type['atom_index_0'].value_counts(normalize=True) < 0.01].index)

    bad_atoms_1 = list(train_type['atom_index_1'].value_counts(normalize=True)[train_type['atom_index_1'].value_counts(normalize=True) < 0.01].index)

    bad_atoms = list(set(bad_atoms_0 + bad_atoms_1))

    train_type = train_type.loc[(train_type['atom_index_0'].isin(bad_atoms_0) == False) & (train_type['atom_index_1'].isin(bad_atoms_1) == False)]

    

    G = nx.from_pandas_edgelist(train_type, 'atom_index_0', 'atom_index_1', ['scalar_coupling_constant'])

    plt.subplot(2, 4, i + 1);

    nx.draw(G, with_labels=True);

    plt.title(f'Graph for type {t}')









from sklearn import preprocessing #import LableEncoder



for f in ['atom_0', 'atom_1', 'type_0', 'type_1', 'type']:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(train[f].values) + list(test[f].values))

    train[f] = lbl.transform(list(train[f].values))

    test[f] = lbl.transform(list(test[f].values))



train.head()
n_fold = 5

from sklearn.model_selection import KFold

folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

params = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'regression',

          'max_depth': 11,

          'learning_rate': 0.2,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.3,

          'colsample_bytree': 1.0

         }
import lightgbm as lgb
#for t,val in folds.split(train):

from sklearn.model_selection import train_test_split

t,val=train_test_split(train,test_size=0.2)

X = t.drop(['id', 'molecule_name', 'scalar_coupling_constant'], axis=1)

Y = t['scalar_coupling_constant']

lgb_train = lgb.Dataset(X, Y)

X_val = val.drop(['id', 'molecule_name','scalar_coupling_constant'], axis=1)

Y_val = val['scalar_coupling_constant']

lgb_eval = lgb.Dataset(X_val, Y_val, reference=lgb_train)

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=20,

                valid_sets=lgb_eval,

                early_stopping_rounds=5)



    
pred=gbm.predict(X_val)

#print(pred)

#print(Y_val)

maes = (Y_val-pred).abs()

print(np.log(maes.map(lambda x:x).mean()))

    
# scores = []

# feature_importance = pd.DataFrame()

# fold_importance = pd.DataFrame()

#             fold_importance["feature"] = columns

#             fold_importance["importance"] = model.feature_importances_

#             fold_importance["fold"] = fold_n + 1

#             feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

import seaborn as sns

scores=gbm.feature_importance()

features=gbm.feature_name()

feature_importance = pd.DataFrame()

feature_importance["feature"] = features#train.columns

feature_importance["importance"] = scores#gbm.feature_importance()

#f=sns.load_dataset(feature_importance).sort_value(by="importance", ascending=False)

feature_importance.sort_values(by=["importance"], ascending=False, inplace=True)

plt.figure(figsize=(16, 12))

sns.barplot(x="importance", y="feature", data=feature_importance)
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
df_val = pd.DataFrame({"type":X_val["type"]})

df_val['scalar_coupling_constant'] = Y_val
def metric(df, preds):

    df['diff'] = (df['scalar_coupling_constant'] - preds).abs()

    return np.log(df.groupby('type')['diff'].mean().map(lambda x: max(x, 1e-9))).mean()
def evaluate_metric(params):

    

    model_lgb = lgb.train(params, lgb_train, 500, 

                          valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=20, 

                          verbose_eval=500)



    pred = model_lgb.predict(X_val)

    #pred=gbm.predict(X_val)

    #print(pred)

    #print(Y_val)

    #maes = (Y_val-pred).abs()

    #print(np.log(maes.map(lambda x:x).mean()))



    score = metric(df_val, pred)

    

    print(score)

 

    return {

        'loss': score,

        'status': STATUS_OK,

        'stats_running': STATUS_RUNNING

    }
hyper_space = {'objective': 'regression',

               'metric':'mae',

               'boosting':'gbdt',

               #'n_estimators': hp.choice('n_estimators', [25, 40, 50, 75, 100, 250, 500]),

               'max_depth':  hp.choice('max_depth', [5, 8, 10, 12, 15]),

               'num_leaves': hp.choice('num_leaves', [100, 250, 500, 650, 750, 1000,1300]),

               'subsample': hp.choice('subsample', [.3, .5, .7, .8, 1]),

               'colsample_bytree': hp.choice('colsample_bytree', [ .6, .7, .8, .9, 1]),

               'learning_rate': hp.choice('learning_rate', [.1, .2, .3]),

               'reg_alpha': hp.choice('reg_alpha', [.1, .2, .3, .4, .5, .6]),

               'reg_lambda':  hp.choice('reg_lambda', [.1, .2, .3, .4, .5, .6]),               

               'min_child_samples': hp.choice('min_child_samples', [20, 45, 70, 100])}
from functools import partial

trials = Trials()



# Set algoritm parameters

algo = partial(tpe.suggest, 

               n_startup_jobs=-1)



# Seting the number of Evals

MAX_EVALS= 5

#MAX_EVALS=15

# Fit Tree Parzen Estimator

best_vals = fmin(evaluate_metric, space=hyper_space, verbose=1,

                 algo=algo, max_evals=MAX_EVALS, trials=trials)



# Print best parameters

best_params = space_eval(hyper_space, best_vals)
best_params
gbm = lgb.train(best_params,

                lgb_train,

                num_boost_round=20,

                valid_sets=lgb_eval,

                early_stopping_rounds=5)

pred=gbm.predict(X_val)

#print(pred)

#print(Y_val)

maes = (Y_val-pred).abs()

print(np.log(maes.map(lambda x:x).mean()))
test.drop(['molecule_name'], axis=1,inplace=True)
#test.drop(['molecule_name'], axis=1,inplace=True)

y_preds = gbm.predict(test)

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='id')

predictions = sample_submission.copy()

predictions['scalar_coupling_constant'] = y_preds[0]

predictions.to_csv('submission.csv')
