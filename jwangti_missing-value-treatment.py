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
melb_data = pd.read_csv('../input/melb_data.csv')
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
melb_target = melb_data.Price
melb_predictor = melb_data.drop(['Price'], axis = 1)
melb_numeric_predictor = melb_predictor.select_dtypes(exclude = ['object'])
X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictor, melb_target, train_size = 0.7, test_size = 0.3, random_state = 0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return mean_absolute_error(prediction, y_test)

#1.Drop
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
reduce_X_train = X_train.drop(cols_with_missing, axis = 1)
reduce_X_test = X_test.drop(cols_with_missing, axis = 1)
mae_red = score_dataset(reduce_X_train, reduce_X_test, y_train, y_test)
#2.Inpute
my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
mae_imp = score_dataset(imputed_X_train, imputed_X_test, y_train, y_test)
#3.Inpute Extra
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

my_imputer_plus = SimpleImputer()
imputed_X_train_plus = my_imputer_plus.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer_plus.transform(imputed_X_test_plus)

mae_imp_pls = score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test)

print(mae_red)
print(mae_imp)
print(mae_imp_pls)
