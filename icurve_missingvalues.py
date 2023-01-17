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
data = pd.read_csv('../input/melb_data.csv')



y = data.Price

X = data.drop('Price',axis=1)



X = X.select_dtypes(exclude=['object'])

X.head()
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state = 4)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



def score_dataset(X_train,X_valid,y_train,y_valid):

    model = RandomForestRegressor()

    model.fit(X_train,y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid,preds)
col_with_missing = [col for col in X_train.columns

                   if X_train[col].isnull().any()]



reduce_X_train = X_train.drop(col_with_missing,axis=1)

reduce_X_valid = X_valid.drop(col_with_missing,axis=1)



print('MAE 1(DORP):')

print(score_dataset(reduce_X_train,reduce_X_valid,y_train,y_valid))
from sklearn.impute import SimpleImputer



my_impute = SimpleImputer()

impute_X_train = pd.DataFrame(my_impute.fit_transform(X_train))

impute_X_valid = pd.DataFrame(my_impute.transform(X_valid))

impute_X_train.columns = X_train.columns

impute_X_valid.columns = X_valid.columns



print('MAE 2(impute):')

print(score_dataset(impute_X_train,impute_X_valid,y_train,y_valid))
my_impute = SimpleImputer()

X_train_plus = X_train.copy()

X_valid_plus = X_valid.copy()

for col in col_with_missing:

    X_train_plus[col+'_with_missing']=X_train[col].isnull()

    X_valid_plus[col+'_with_missing']=X_valid[col].isnull()

    pass



imputed_X_train = pd.DataFrame(my_impute.fit_transform(X_train_plus))

imputed_X_valid = pd.DataFrame(my_impute.transform(X_valid_plus))

imputed_X_train.columns = X_train_plus.columns

imputed_X_valid.columns = X_valid_plus.columns



print('MAE 3(E impute):')

print(score_dataset(imputed_X_train,imputed_X_valid,y_train,y_valid))