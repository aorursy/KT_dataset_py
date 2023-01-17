import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

structures = pd.read_csv('../input/structures.csv')

scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')



print('Train dataset shape is -> rows: {} cols:{}'.format(train.shape[0],train.shape[1]))

print('Test dataset shape is  -> rows: {} cols:{}'.format(test.shape[0],test.shape[1]))

print('Sub dataset shape is  -> rows: {} cols:{}'.format(sub.shape[0],sub.shape[1]))

print('Structures dataset shape is  -> rows: {} cols:{}'.format(structures.shape[0],structures.shape[1]))

print('Scalar_coupling_contributions dataset shape is  -> rows: {} cols:{}'.format(scalar_coupling_contributions.shape[0],

                                                                                   scalar_coupling_contributions.shape[1]))
train = pd.merge(train, scalar_coupling_contributions, how = 'left',

                  left_on  = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],

                  right_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
print('The training set has shape {}'.format(train.shape))

print('The test set has shape {}'.format(test.shape))
# Distribution of the target

train['scalar_coupling_constant'].plot(kind='hist', figsize=(20, 5), bins=1000, title='Distribution of the target scalar coupling constant')

plt.show()
# Number of of atoms in molecule

fig, ax = plt.subplots(1, 2)

train.groupby('molecule_name').count().sort_values('id')['id'].plot(kind='hist',

                                                                       bins=25,

                                                                      figsize=(20, 5),

                                                                      title='# of Atoms in Molecule (Train Set)',

                                                                      ax=ax[0])

test.groupby('molecule_name').count().sort_values('id')['id'].plot(kind='hist',

                                                                       bins=25,

                                                                      figsize=(20, 5),

                                                                      title='# of Atoms in Molecule (Test Set)',

                                                                     ax=ax[1])

plt.show()
## Additional Data

mst = pd.read_csv('../input/magnetic_shielding_tensors.csv')

mul = pd.read_csv('../input/mulliken_charges.csv')

pote = pd.read_csv('../input/potential_energy.csv')

scc = pd.read_csv('../input/scalar_coupling_contributions.csv')
mst.head(3)
mul.head(3)
# Plot the distribution of mulliken_charges

mul['mulliken_charge'].plot(kind='hist', figsize=(15, 5), bins=500, title='Distribution of Mulliken Charges')

plt.show()
# Plot the distribution of potential_energy

pote['potential_energy'].plot(kind='hist',

                              figsize=(15, 5),

                              bins=500,

                              title='Distribution of Potential Energy',

                              color='b')

plt.show()
scalar_coupling_contributions.head()
scc.groupby('type').count()['molecule_name'].sort_values().plot(kind='barh',

                                                                color='red',

                                                               figsize=(15, 5),

                                                               title='Count of Coupling Type in Train Set')

plt.show()
fig, ax = plt.subplots(2, 2, figsize=(20, 10))

scc['fc'].plot(kind='hist', ax=ax.flat[0], bins=500, title='Fermi Contact contribution', color="red")

scc['sd'].plot(kind='hist', ax=ax.flat[1], bins=500, title='Spin-dipolar contribution', color="blue")

scc['pso'].plot(kind='hist', ax=ax.flat[2], bins=500, title='Paramagnetic spin-orbit contribution', color="green")

scc['dso'].plot(kind='hist', ax=ax.flat[3], bins=500, title='Diamagnetic spin-orbit contribution', color="orange")

plt.show()
import seaborn as sns

sns.pairplot(data=train.sample(5000), hue='type', vars=['fc','sd','pso','dso','scalar_coupling_constant'])

plt.show()
## Target vs atom count

atom_count_dict = structures.groupby('molecule_name').count()['atom_index'].to_dict()
train['atom_count'] = train['molecule_name'].map(atom_count_dict)

test['atom_count'] = test['molecule_name'].map(atom_count_dict)
# Map the atom structure data into train and test files



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
# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark

train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
# make categorical variables

atom_map = {'H': 0,

            'C': 1,

            'N': 2}

train['atom_0_cat'] = train['atom_0'].map(atom_map).astype('int')

train['atom_1_cat'] = train['atom_1'].map(atom_map).astype('int')

test['atom_0_cat'] = test['atom_0'].map(atom_map).astype('int')

test['atom_1_cat'] = test['atom_1'].map(atom_map).astype('int')
# One Hot Encode the Type

train = pd.concat([train, pd.get_dummies(train['type'])], axis=1)

test = pd.concat([test, pd.get_dummies(test['type'])], axis=1)
train.head(5)
train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')

test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')
# Configurables

FEATURES = ['atom_index_0', 'atom_index_1',

            'atom_0_cat',

            'x_0', 'y_0', 'z_0',

            'atom_1_cat', 

            'x_1', 'y_1', 'z_1', 'dist', 'dist_to_type_mean',

            'atom_count',

            '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN'

           ]

TARGET = 'scalar_coupling_constant'

CAT_FEATS = ['atom_0','atom_1']

N_ESTIMATORS = 2000

VERBOSE = 500

EARLY_STOPPING_ROUNDS = 200

RANDOM_STATE = 529



X = train[FEATURES]

X_test = test[FEATURES]

y = train[TARGET]
from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error

import lightgbm as lgb



lgb_params = {'num_leaves': 128,

              'min_child_samples': 64,

              'objective': 'regression',

              'max_depth': 6,

              'learning_rate': 0.1,

              "boosting_type": "gbdt",

              "subsample_freq": 1,

              "subsample": 0.9,

              "bagging_seed": 11,

              "metric": 'mae',

              "verbosity": -1,

              'reg_alpha': 0.1,

              'reg_lambda': 0.4,

              'colsample_bytree': 1.0

         }

RUN_LGB = True

if RUN_LGB:

    n_fold = 5

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)



    # Setup arrays for storing results

    oof = np.zeros(len(X))

    prediction = np.zeros(len(X_test))

    scores = []

    feature_importance = pd.DataFrame()



    # Train the model

    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X)):

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]

        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMRegressor(**lgb_params, n_estimators = N_ESTIMATORS, n_jobs = -1)

        model.fit(X_train, y_train,

                  eval_set=[(X_train, y_train), (X_valid, y_valid)],

                  eval_metric='mae',

                  verbose=VERBOSE,

                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)



        y_pred_valid = model.predict(X_valid)

        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        # feature importance

        fold_importance = pd.DataFrame()

        fold_importance["feature"] = FEATURES

        fold_importance["importance"] = model.feature_importances_

        fold_importance["fold"] = fold_n + 1

        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



        prediction /= folds.n_splits

        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

        oof[valid_idx] = y_pred_valid.reshape(-1,)

        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred
if RUN_LGB:

    # Plot feature importance as done in https://www.kaggle.com/artgor/artgor-utils

    feature_importance["importance"] /= folds.n_splits

    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

        by="importance", ascending=False)[:50].index



    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



    plt.figure(figsize=(15, 20));

    ax = sns.barplot(x="importance",

                y="feature",

                hue='fold',

                data=best_features.sort_values(by="importance", ascending=False));

    plt.title('LGB Features (avg over folds)');
prediction_final = model.predict(X_test)
prediction_final
test.shape
prediction_final.shape
sub["scalar_coupling_constant"] = prediction_final

sub.to_csv('submission.csv', index=False)

sub.head()