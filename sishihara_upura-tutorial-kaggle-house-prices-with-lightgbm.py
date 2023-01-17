import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import lightgbm as lgb





def label_encoding(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict):

    """

    col_definition: encode_col

    """

    n_train = len(train)

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in col_definition['encode_col']:

        try:

            lbl = preprocessing.LabelEncoder()

            train[f] = lbl.fit_transform(list(train[f].values))

        except:

            print(f)

    test = train[n_train:].reset_index(drop=True)

    train = train[:n_train]

    return train, test
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
unique_cols = list(train.columns[train.nunique() == 1])

print(unique_cols)
duplicated_cols = list(train.columns[train.T.duplicated()])

print(duplicated_cols)
categorical_cols = list(train.select_dtypes(include=['object']).columns)

print(categorical_cols)
train[categorical_cols].head()
train, test = label_encoding(train, test, col_definition={'encode_col': categorical_cols})
train[categorical_cols].head()
X_train = train.drop(['Id', 'SalePrice'], axis=1)

y_train = np.log1p(train['SalePrice'])

X_test = test.drop(['Id', 'SalePrice'], axis=1)
y_preds = []

models = []

oof_train = np.zeros((len(X_train),))

cv = KFold(n_splits=5, shuffle=True, random_state=0)



params = {

    'num_leaves': 24,

    'max_depth': 6,

    'objective': 'regression',

    'metric': 'rmse',

    'learning_rate': 0.05

}



for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    lgb_train = lgb.Dataset(X_tr,

                            y_tr,

                            categorical_feature=categorical_cols)



    lgb_eval = lgb.Dataset(X_val,

                           y_val,

                           reference=lgb_train,

                           categorical_feature=categorical_cols)



    model = lgb.train(params,

                      lgb_train,

                      valid_sets=[lgb_train, lgb_eval],

                      verbose_eval=10,

                      num_boost_round=1000,

                      early_stopping_rounds=10)





    oof_train[valid_index] = model.predict(X_val,

                                           num_iteration=model.best_iteration)

    y_pred = model.predict(X_test,

                           num_iteration=model.best_iteration)



    y_preds.append(y_pred)

    models.append(model)
print(f'CV: {np.sqrt(mean_squared_error(y_train, oof_train))}')
y_sub = sum(y_preds) / len(y_preds)

y_sub = np.expm1(y_sub)

sub['SalePrice'] = y_sub

sub.to_csv('submission.csv', index=False)

sub.head()