# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data = pd.read_csv('../input/train.csv')
gender = pd.read_csv('../input/gender_submission.csv')
test_data = pd.read_csv('../input/test.csv')
target = data.Survived

#train-test splitting after dropping useless data
data = data.drop(['PassengerId', 'Survived'], axis=1)
train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.2)
upd_test_data = test_data.drop(['PassengerId'], axis=1)

#one hot encoding
ohe_tr = pd.get_dummies(train_x)
ohe_val = pd.get_dummies(val_x)
ohe_test_data = pd.get_dummies(upd_test_data)

#imputing + assigning columns
my_imputer = Imputer()
ohe_imp_tr = pd.DataFrame(my_imputer.fit_transform(ohe_tr))
ohe_imp_tr.columns = ohe_tr.columns
ohe_imp_val = pd.DataFrame(my_imputer.fit_transform(ohe_val))
ohe_imp_val.columns = ohe_val.columns
ohe_test_pred = pd.DataFrame(my_imputer.fit_transform(ohe_test_data))
ohe_test_pred.columns = ohe_test_data.columns

#joining matching columns
inter_train, inter_val = ohe_imp_tr.align(ohe_imp_val, join = 'inner', axis=1)
final_train, inter_test = inter_train.align(ohe_test_pred, join = 'inner', axis=1)
final_val, final_test = inter_val.align(inter_test, join='inner',axis=1)

#creating model
my_model = XGBRegressor(n_estimators=1000,learning_rate=0.05, objective= 'binary:logistic')
my_model.fit(final_train, train_y, early_stopping_rounds=5, 
             eval_set=[(final_val, val_y)], verbose=False)

pred_survival = my_model.predict(final_test)
pred_survival = np.around(pred_survival)
pred_survival = pred_survival.astype(int)

print(pred_survival)
my_submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': pred_survival})
my_submission.to_csv('submission.csv', index=False)


# Any results you write to the current directory are saved as output.
