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
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

target = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train.head()
target.head()
# Проверил упорядоченность записей в train и target

np.all(train['sig_id'] == target['sig_id'])
try:

    train.drop(columns=['sig_id'], inplace=True)

except Exception:

    print('Column already deleted!')



try:

    target.drop(columns=['sig_id'], inplace=True)

except Exception:

    print('Column already deleted!')
# Некоторые признаки нужно перевести в категориальные

train.info(verbose=True)
target.info(verbose=True)
from sklearn.preprocessing import OneHotEncoder

cat_columns = ['cp_type', 'cp_dose']

cat_train = train[cat_columns]

enc = OneHotEncoder(categories='auto', drop='first')

cat_train = enc.fit_transform(cat_train).toarray()

cat_train = pd.DataFrame(cat_train)

not_cat_cols = train.columns[train.columns.isin(cat_columns) != True]

not_cat_train = train[not_cat_cols]

train_ohe = pd.concat([cat_train, not_cat_train], axis=1, ignore_index=True)

train_ohe
X, y = train_ohe, target

print(X.shape, y.shape)
import statsmodels.api as sm

import patsy as pt

import sklearn.linear_model as lm



# создаем пустую модель

skm = lm.LinearRegression()

# запускаем расчет параметров для указанных данных

skm.fit(X, y)

pred = skm.predict(X)
from sklearn.metrics import mean_squared_error as mse

loss = mse(y, pred, multioutput='raw_values')

for i, col in enumerate(y.columns):

    print('{}{}{}{:.5f}'.format('Name: ',col,', MSE loss: ', loss[i]), sep='')
cat_test = test[cat_columns]

cat_test = enc.transform(cat_test).toarray()

cat_test = pd.DataFrame(cat_test)

not_cat_cols = train.columns[train.columns.isin(cat_columns) != True]

not_cat_test = test[not_cat_cols]

test_ohe = pd.concat([cat_test, not_cat_test], axis=1, ignore_index=True)

test_ohe
pred_test = skm.predict(test_ohe)

res = pd.concat([test[['sig_id']], pd.DataFrame(pred_test, columns=target.columns)], axis=1)
res.to_csv("submission.csv", index=None)

res.head()