# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.model_selection import KFold, train_test_split

import lightgbm as lgb

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

sub = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head()
test.head()

test.info()
sub.head()
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
categorical_cols = list(train.select_dtypes(include=['object']).columns)
train, test = label_encoding(train, test, col_definition={'encode_col': categorical_cols})
X = train.drop(['Id', 'SalePrice'], axis=1)

y = np.log1p(train['SalePrice'])

X_Test = test.drop(['Id','SalePrice'], axis=1)
X.head()
y.head()
X_Test.head()


X_train, X_test, y_train, y_test = train_test_split(X, y)





params = {

    'num_leaves': 24,

    'max_depth': 6,

    'objective': 'regression',

    'metric': 'rmse',

    'learning_rate': 0.05

}

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_cols)

lgb_eval = lgb.Dataset(X_test,

                           y_test,

                           reference=lgb_train,

                           categorical_feature=categorical_cols)

model = lgb.train(params,

                  lgb_train,

                  valid_sets=[lgb_train, lgb_eval],

                  verbose_eval=10,

                          num_boost_round=1000,

                          early_stopping_rounds=10)



y_pred = np.expm1(model.predict(X_Test,num_iteration=model.best_iteration))

y_pred = np.expm1(model.predict(X_Test,num_iteration=model.best_iteration))

sub["SalePrice"] = y_pred

sub.to_csv("submission.csv", index=False)

sub.head()