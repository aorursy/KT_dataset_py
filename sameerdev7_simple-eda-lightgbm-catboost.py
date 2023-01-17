import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pylab as plt

import seaborn as sns

from sklearn import metrics

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')

color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')

submission.head()
train.head()
test.head()
train.shape
test.shape
structures = pd.read_csv('../input/structures.csv')

display(structures.head())
# Distribution of the target

train['scalar_coupling_constant'].plot(kind='hist', figsize=(20, 5), bins=1000, title='Distribution of the target scalar coupling constant')

plt.show()
# Number of of atoms in molecule

fig, ax = plt.subplots(1, 2)

train.groupby('molecule_name').count().sort_values('id')['id'].plot(kind='hist',

                                                                       bins=25,

                                                                       color=color_pal[6],

                                                                      figsize=(20, 5),

                                                                      title='# of Atoms in Molecule (Train Set)',

                                                                      ax=ax[0])

test.groupby('molecule_name').count().sort_values('id')['id'].plot(kind='hist',

                                                                       bins=25,

                                                                       color=color_pal[2],

                                                                      figsize=(20, 5),

                                                                      title='# of Atoms in Molecule (Test Set)',

                                                                     ax=ax[1])

plt.show()
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
%%time

# Engineer a single feature: distance vector between atoms



train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
train.head()
# Label Encoding

train = pd.get_dummies(data=train, columns=['type','atom_0','atom_1'])

test = pd.get_dummies(data=test, columns=['type','atom_0','atom_1'])
atom_count_dict = structures.groupby('molecule_name').count()['atom_index'].to_dict()



train['atom_count'] = train['molecule_name'].map(atom_count_dict)

test['atom_count'] = test['molecule_name'].map(atom_count_dict)
train.dtypes
train.head()
labels = train['scalar_coupling_constant'].values

train_data = train.drop(['id','molecule_name','scalar_coupling_constant'], axis=1)

test_data = test.drop(['id','molecule_name'], axis=1)
train_data.head()
%%time

#splitting into train and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(train_data, labels, test_size=0.1, random_state=47)
%%time

#Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(pd.concat([train_data,test_data]))

x_train = sc.transform(x_train)

x_test = sc.transform(x_test)

test_data = sc.transform(test_data)
#Parameters for LightGBM Model

params_lgb = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'regression',

          'max_depth': 9,

          'learning_rate': 0.1,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.9,

          "bagging_seed": 47,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1302650970728192,

          'reg_lambda': 0.3603427518866501,

          'colsample_bytree': 1.0,

          'n_estimators':10000

         }
%%time

#LightGBM Model

import lightgbm as lgb

lgtrain = lgb.Dataset(x_train, label=y_train)

lgval = lgb.Dataset(x_test, label=y_test)

model_lgb = lgb.train(params_lgb, lgtrain, 5000, 

                  valid_sets=[lgtrain, lgval], 

                  verbose_eval=500)
#Submission for LightGBM Model

lgb_pred = model_lgb.predict(test_data)

submission['scalar_coupling_constant']=lgb_pred

submission.to_csv('lgb_model.csv',index=False)
%%time

#CatBoost Model

from catboost import CatBoostRegressor



model_cat = CatBoostRegressor(iterations=5000,

                             learning_rate=0.03,

                             depth=2,

                             eval_metric='MAE',

                             random_seed = 47,

                             od_wait=5000)



model_cat.fit(x_train,y_train, verbose=500)
#Submission for CatBoost Model

cat_pred = model_cat.predict(test_data)

submission['scalar_coupling_constant']=cat_pred

submission.to_csv('cat_model.csv',index=False)