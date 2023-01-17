import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgbm

from scipy import sparse

from sklearn import metrics

from sklearn import naive_bayes

from sklearn import linear_model

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
root = "../input/cat-in-the-dat"



train = pd.read_csv(os.path.join(root, 'train.csv'))

test = pd.read_csv(os.path.join(root, 'test.csv'))

target = train['target']

train = train.drop(columns='target')

both = pd.concat([train, test], sort=False)

print(train.shape)

print(test.shape)

print(both.shape)

both.head()
both['bin_3'] = both['bin_3'].map({"T": 1, "F": 0})

both['bin_4'] = both['bin_4'].map({"Y": 1, "N": 0})

both[['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']].head()
onehot_encoded_vars = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

dummies = pd.get_dummies(both[onehot_encoded_vars], prefix=onehot_encoded_vars, drop_first=True, sparse=True).sparse.to_coo()

both = both.drop(columns=onehot_encoded_vars)

print(dummies)
both['ord_1'] = both['ord_1'].map({

    "Novice": 0,

    "Contributor": 1,

    "Expert": 2,

    "Master": 3,

    "Grandmaster": 4

})

both['ord_2'] = both['ord_2'].map({

    "Freezing": 0,

    "Cold": 1,

    "Warm": 2,

    "Hot": 3,

    "Boiling Hot": 4,

    "Lava Hot": 5

})

both[['ord_0', 'ord_1', 'ord_2']].head()
ordinal_encoded_vars = ['ord_3', 'ord_4', 'ord_5']

ordinal_encoder = preprocessing.OrdinalEncoder()

both[ordinal_encoded_vars] = ordinal_encoder.fit_transform(both[ordinal_encoded_vars], y = target)

both[ordinal_encoded_vars].head()
both['month'] = both['month'].multiply(2 * np.pi / 12)

both['month_sin'] = np.sin(both['month'])

both['month_cos'] = np.cos(both['month'])

both.drop(columns='month')



both['day'] = train['day'].multiply(2 * np.pi / 7)

both['day_sin'] = np.sin(both['day'])

both['day_cos'] = np.cos(both['day'])

both.drop(columns='day')



both[['month_sin', 'month_cos', 'day_sin', 'day_cos']].head()
D = sparse.hstack([dummies, both.astype('float32')]).tocsr()

train_data = D[:train.shape[0]]

print(train_data)
%%time



params = {

    'objective' :'binary',

    'learning_rate' : 0.005,

    'num_leaves' : 80,

    'metric': 'auc',

    'min_data_in_leaf': 500,

    'feature_fraction': 0.64, 

    'bagging_fraction': 0.8, 

    'bagging_freq': 1,

    'boosting_type' : 'gbdt'

}



X_train, X_valid, Y_train, Y_valid = train_test_split(train_data,  target, random_state=7, test_size=0.20)

d_train = lgbm.Dataset(X_train, Y_train)

d_valid = lgbm.Dataset(X_valid, Y_valid)

model = lgbm.train(params, d_train, 10000, valid_sets=[d_valid], verbose_eval=100, early_stopping_rounds=500)
test_data = D[train.shape[0]:]

predictions = model.predict(test_data)

submit = pd.concat([test['id'], pd.Series(predictions).rename('target')], axis=1)

submit.to_csv('submission.csv', index=False, header=True)

submit