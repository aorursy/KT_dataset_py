# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

from sklearn.model_selection import train_test_split

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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
X = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)
print('Train data: {}'.format(X.shape))

print('Train data targets: {}'.format(len(y)))

print('---------------------------------------------------------')

print('Test data: {}'.format(X_test.shape))

print('---------------------------------------------------------', '\n')

X_all = pd.concat([X,X_test],)
X_all.isnull().sum()
plt.figure(figsize=(25,25))

sns.heatmap(X_all.isnull())
missing_cols = X_all.isnull().sum()/X_all.shape[0]*100
drop_col = missing_cols[missing_cols>10].keys()

drop_col
X_all.drop(X_all[drop_col],axis=1,inplace=True)
missing_columns= missing_cols[missing_cols>10].keys()

missing_columns
cat_col = X_all[missing_columns].select_dtypes(include='object')
num_col = X_all[missing_columns].select_dtypes(exclude='object')
object_missing = [x for x in missing_columns if x not in num_col]

none_missing = [x for x in cat_col if x not in cat_col]
for col in num_col:

    X_all[col] = X_all[col].fillna(0)



for col in cat_col:

    X_all[col] = X_all[col].fillna(X_all[col].mode()[0])



for col in none_missing:

    X_all[col] = X_all[col].fillna('none')

    

# Check the final result

X_all.isna().sum()[X_all.isna().sum() > 0]
# One-Hot encoding

print(X_all.shape)

X_all = pd.get_dummies(X_all).reset_index(drop=True)

print(X_all.shape)
numeric_features = [col for col in X_all.columns if X_all[col].dtype in ['float64', 'int64']]



mean = X_all[numeric_features].mean(axis=0)

std = X_all[numeric_features].std(axis=0)



X_all[numeric_features] -= mean # centering

X_all[numeric_features] /= std # scaling
# Return train and test sets 

X_new = X_all.iloc[:1460, :]

X_test_new = X_all.iloc[1460: , :]



# Create data sets for training (80%) and validation (20%)

X_train, X_valid, y_train, y_valid = train_test_split(X_new, y, train_size=0.8, test_size=0.2, random_state=0)
# Return train and test sets 

X_new = X_all.iloc[:1460, :]

X_test_new = X_all.iloc[1460: , :]



# Create data sets for training (80%) and validation (20%)

X_train, X_valid, y_train, y_valid = train_test_split(X_new, y, train_size=0.8, test_size=0.2, random_state=0)
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# The best model

params = {'n_estimators': 4000,

          'max_depth': 6,

          'min_child_weight': 3,

          'learning_rate': 0.02,

          'subsample': 0.7,

          'random_state': 0}



model = XGBRegressor(**params)



model.fit(X_train, y_train, verbose=False)



preds = model.predict(X_valid)

print('Valid MAE of the best model: {}'.format(mean_absolute_error(preds, y_valid)))
predictions = model.predict(X_test_new)



output = pd.DataFrame({'Id': X_test.Id,

                       'SalePrice': predictions})

output.to_csv('submission.csv', index=False)