# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split 

import lightgbm as lgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv(r"/kaggle/input/train-test-universe/Glove_train_universe.csv")

test = pd.read_csv(r"/kaggle/input/train-test-universe/Glove_test_universe.csv")
train.head()
# train_final = train_final.drop('train', axis = 1)

# test_final = test_final.drop(['train','price'], axis = 1)
y = np.log(train['price'].values+1)

X = np.array(train.drop(['price','train_id'], axis=1))
predictors = [name for name in train.columns if name not in ['price','train_id']]

print(len(predictors))

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=0)

d_train = lgb.Dataset(x_train, label=y_train, feature_name=predictors,free_raw_data=False)

d_valid = lgb.Dataset(x_valid, label=y_valid, feature_name=predictors,free_raw_data=False)

watchlist = [d_train, d_valid]
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'max_depth': 10, 

    'num_leaves': 100,

    'learning_rate': 0.08,

    'verbose': 0, 

    'early_stopping_round': 15}



n_estimators = 15000
model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)
test.head()
test['price'] = np.exp(model.predict(test[predictors]))
test.head(10)
np.max(test.price)
submission = pd.DataFrame(test['train_id'])

submission['price'] = test.price

submission.head()
submission.to_csv("submission.csv", index = False)