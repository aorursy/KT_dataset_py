# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

X_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')







# Remove rows with missing target, separate target from predictors

#X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)



#cat = ['YearBuilt', 'GarageYrBlt', 'MoSold', 'YrSold', 'YearRemodAdd']

X['MoSold'] = X['MoSold'].astype('object')

X_test['MoSold'] = X_test['MoSold'].astype('object')
X.shape
X['LotArea'].mean()
# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
M = X_train.isnull().sum(axis = 0)

M[M>50].plot(kind = 'barh')

M
col_remov = list(M[M>500].index)

col_remov
X_train.drop(col_remov, axis = 1, inplace = True)

X_valid.drop(col_remov, axis = 1, inplace = True)

X_test.drop(col_remov, axis = 1, inplace = True)
X_train.shape
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error
X_train.select_dtypes(exclude=['object']).columns
X_train.select_dtypes(include=['object']).columns


# Preprocessing for categorical data

numerical_cols = list(X_train.select_dtypes(exclude=['object']).columns)

categorical_cols = list(X_train.select_dtypes(include=['object']).columns)
len(numerical_cols)
import numpy as np

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=0)

imputer.fit(X_train[numerical_cols].values)

X_train_mat = imputer.transform(X_train[numerical_cols].values)

X_valid_mat = imputer.transform(X_valid[numerical_cols].values)

X_test_mat = imputer.transform(X_test[numerical_cols].values)
categorical_transformer = SimpleImputer(strategy='most_frequent')

categorical_transformer.fit(X_train[categorical_cols])

X_train_cat = categorical_transformer.transform(X_train[categorical_cols].values)

X_valid_cat = categorical_transformer.transform(X_valid[categorical_cols].values)

X_test_cat = categorical_transformer.transform(X_test[categorical_cols].values) 

train_X = np.hstack((X_train_mat, X_train_cat))

valid_X = np.hstack((X_valid_mat, X_valid_cat))

test_X = np.hstack((X_test_mat, X_test_cat))
train_X
train_X_df = pd.DataFrame(train_X, columns = numerical_cols + categorical_cols, index = X_train.index)

valid_X_df = pd.DataFrame(valid_X, columns = numerical_cols + categorical_cols, index = X_valid.index)

test_X_df = pd.DataFrame(test_X, columns = numerical_cols + categorical_cols, index = X_test.index)

train_X_df.head()
train_X_df.isnull().sum().sum()
embed_cols=[i for i in categorical_cols]



for i in embed_cols:

    print(i,train_X_df[i].nunique())
date = train_X_df[['YearBuilt', 'GarageYrBlt', 'MoSold', 'YrSold', 'YearRemodAdd']]

date.head()
# get a look on the old of house

date_min_max = pd.concat([date.min(axis = 0), date.max(axis = 0)],axis = 1)

date_min_max.columns = ['Min_date', 'Max_date']

date_min_max
train_X_df.dtypes
# categoricals features

for col in categorical_cols:

    train_X_df[col] = train_X_df[col].astype(str)

    valid_X_df[col] = valid_X_df[col].astype(str)

    test_X_df[col] = test_X_df[col].astype(str)



#importing library and building model

from catboost import CatBoostRegressor

model=CatBoostRegressor(iterations=1000, learning_rate=0.05,  loss_function='RMSE', logging_level='Silent')
model.fit(train_X_df,y_train,cat_features=categorical_cols,eval_set=(valid_X_df,y_valid),plot=True)
from sklearn.metrics import mean_absolute_error



predictions = model.predict(valid_X_df)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
preds_test = model.predict(test_X_df)
# Save test predictions to file

output = pd.DataFrame({'Id': test_X_df.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)