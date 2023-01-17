# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

import lightgbm as lgb
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission.head()
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train.head()
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test.head()
print('=========== train infomation ===========')

train.info()

print('\n\n=========== test infomation ===========')

test.info()
data = pd.concat([train, test])

data.shape
cat_cols = data.loc[:,data.dtypes == 'object'].columns

data.loc[:,cat_cols].head()
cat_data = pd.get_dummies(data.loc[:, cat_cols], drop_first=True)

cat_data.head()
num_data = data.loc[:,data.dtypes != 'object'].drop('Id', axis=1)

num_data.head()
log_num_data = np.log1p(num_data)

log_num_data.head()
skewness = pd.concat([num_data.apply(lambda x: skew(x.dropna())),

                      log_num_data.apply(lambda x: skew(x.dropna()))],

                     axis=1).rename(columns={0:'original', 1:'logarithmization'})

skewness.plot.barh(figsize=(12,10), title='Comparison of skewness of original and logarithmized', width=0.8);
optimized_data = pd.concat([data['Id'], cat_data, log_num_data], axis=1)

optimized_data.head()
train = optimized_data[:train.shape[0]]

test = optimized_data[train.shape[0]:].drop(['Id', 'SalePrice'], axis=1)

X_train = train.drop(['Id', 'SalePrice'], axis=1)

y_train = train['SalePrice']
lgb_train = lgb.Dataset(X_train, y_train)

params = {

        'task' : 'train',

        'boosting_type' : 'gbdt',

        'objective' : 'regression',

        'metric' : {'l2'},

        'num_leaves' : 40,

        'learning_rate' : 0.1,

        'feature_fraction' : 0.9,

        'bagging_fraction' : 0.8,

        'bagging_freq': 5,

        'verbose' : 0

}

gbm = lgb.train(params, lgb_train)



pred = gbm.predict(test)
pred = np.expm1(pred)

# create submission file

results = pd.Series(pred, name='SalePrice')

submission = pd.concat([submission['Id'], results], axis=1)

submission.to_csv('submission.csv', index=False)

submission.head()