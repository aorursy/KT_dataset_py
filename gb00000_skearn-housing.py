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
dd = open('/kaggle/input/home-data-for-ml-course/data_description.txt', 'r')
for i in dd:

    print(i)

dd.close()
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

samp_sub = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV
train
y = train['SalePrice']
test


train.info()
test.info()
y = train.SalePrice
train_X = train.drop('SalePrice', axis = 1)
train_nona = train_X.dropna(axis=1)

test_nona = test.dropna(axis=1)
test_nona.info()
train_nona_test  = train_nona[[i for i in test_nona.columns if i in train_nona.columns]]
test_nona_train = test_nona[[i for i in train_nona_test.columns]]
train_nona_test.info()
train_nona_test.select_dtypes('object').describe()
train_obj_dum = pd.get_dummies(train_nona_test.select_dtypes('object'))

test_obj_dum = pd.get_dummies(test_nona_train.select_dtypes('object'))
train_obj_dum_test = train_obj_dum[[i for i in test_obj_dum.columns if i in train_obj_dum.columns]]

test_obj_dum_train = test_obj_dum[[i for i in train_obj_dum_test.columns]]
train_obj_dum_test
test_obj_dum_train
train_int64 = train_nona_test.drop('Id', axis=1).select_dtypes('int64')

train_id = train_nona_test['Id']

test_int64 = test_nona_train.drop('Id', axis=1).select_dtypes('int64')

test_id = test_nona_train['Id']
train_norm = pd.DataFrame(preprocessing.StandardScaler().fit_transform(train_int64), columns = train_int64.columns)

test_norm = pd.DataFrame(preprocessing.StandardScaler().fit_transform(test_int64), columns = test_int64.columns)

#train_norm = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(train_int64), columns = train_int64.columns)

#test_norm = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(test_int64), columns = test_int64.columns)
train_norm
train_clean = pd.concat([train_norm,train_obj_dum_test],axis=1)

test_clean = pd.concat([test_norm,test_obj_dum_train],axis=1)
train_clean
test_clean
X_train, X_test, y_train, y_test = train_test_split(train_clean, y)
modelrfr = RandomForestRegressor(n_estimators = 200, criterion='mae', random_state=42)

modelrfr.fit(X_train, y_train)

y_predrfr = modelrfr.predict(X_test)

mean_absolute_error(y_test, y_predrfr)
modeldtr = DecisionTreeRegressor(criterion='mae', random_state=42)

modeldtr.fit(X_train, y_train)

y_preddtr = modelrfr.predict(X_test)

mean_absolute_error(y_test, y_preddtr)


modelgbr = GradientBoostingRegressor(n_estimators = 200, criterion='mae', random_state=42)

modelgbr.fit(X_train, y_train)

y_predgbr = modelgbr.predict(X_test)

mean_absolute_error(y_test, y_predgbr)
modelabr = AdaBoostRegressor(n_estimators = 200, random_state=42)

modelabr.fit(X_train, y_train)

y_predabr = modelabr.predict(X_test)

mean_absolute_error(y_test, y_predabr)
modelbr = BaggingRegressor(n_estimators = 200,  random_state=42)

modelbr.fit(X_train, y_train)

y_predbr = modelbr.predict(X_test)

mean_absolute_error(y_test, y_predbr)
modeletr = ExtraTreesRegressor(n_estimators = 200, criterion='mae', random_state=42)

modeletr.fit(X_train, y_train)

y_predetr = modeletr.predict(X_test)

mean_absolute_error(y_test, y_predetr)
modellr = LinearRegression()

modellr.fit(X_train, y_train)

y_predlr = modellr.predict(X_test)

mean_absolute_error(y_test, y_predlr)
test_id.values
modelsub = GradientBoostingRegressor(n_estimators = 200, criterion='mae', random_state=42)

modelsub.fit(train_clean, y)

preds_test = modelrfr.predict(test_clean)
output = pd.DataFrame({'Id': test_id.values,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)


