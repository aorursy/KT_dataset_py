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

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train=pd.get_dummies(train)

test=pd.get_dummies(test)



train=pd.concat(

    [train.loc[:,train.columns.intersection(test.columns)],train[['SalePrice']]],

    axis=1

)
from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

train_set,validation_set=train_test_split(train, test_size=0.3)

X_train=train_set.drop(['Id','SalePrice'],axis=1)

y_train=train_set.loc[:,'SalePrice']

X_validation=validation_set.drop(['Id','SalePrice'],axis=1)

y_validation=validation_set.loc[:,'SalePrice']

imputed_X_train_plus = X_train.copy()

imputed_X_validate_plus = X_validation.copy()

imputed_X_test_plus=test.copy()

cols_with_missing = (col for col in X_train.columns 

                                 if X_train[col].isnull().any())

for col in cols_with_missing:

    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

    imputed_X_validate_plus[col + '_was_missing'] = imputed_X_validate_plus[col].isnull()

    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()



imputed_X_train_plus=imputed_X_train_plus.fillna(imputed_X_train_plus.mean())

imputed_X_validate_plus=imputed_X_validate_plus.fillna(imputed_X_train_plus.mean())

imputed_X_test_plus=imputed_X_test_plus.fillna(imputed_X_train_plus.mean())
from sklearn.model_selection import ParameterGrid

param_grid = {'n_estimators': [50,100,200], 'learning_rate': [0.01,0.1,0.5], 'max_depth':[2,3,4,5]}



best_model=None

best_validation_loss=float('inf')

best_parameter=None



for params in list(ParameterGrid(param_grid)):

    print('n_estimators is {}, learning_rate is {}, max_depth is {}'.format(

        params['n_estimators'],

        params['learning_rate'],

        params['max_depth'],

    ))

    est=GradientBoostingRegressor(

        n_estimators=params['n_estimators'],

        learning_rate=params['learning_rate'],

        max_depth=params['max_depth'],

        random_state=0,

        loss='ls',

    ).fit(imputed_X_train_plus,y_train)

    training_loss=mean_squared_log_error(y_train,est.predict(imputed_X_train_plus))

    validation_loss=mean_squared_log_error(y_validation,est.predict(imputed_X_validate_plus))

    print('validation loss is {:.5f}'.format(validation_loss))

    print('training loss is {:.5f}'.format(training_loss))

    

    if validation_loss<best_validation_loss:

        best_validation_loss=validation_loss

        best_model=est

        best_parameter=params

print(best_validation_loss)

print(best_parameter)
predictions_array=best_model.predict(imputed_X_test_plus.iloc[:,imputed_X_test_plus.columns!='Id'])
result=pd.concat([test[['Id']],pd.DataFrame(predictions_array,columns=['SalePrice'])],axis=1)
result.to_csv('submission.csv',index=False)