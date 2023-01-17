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
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.model_selection import GridSearchCV, KFold, cross_validate

from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Normalizer, OrdinalEncoder, StandardScaler

from sklearn.impute import SimpleImputer
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

samp_sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
dd = open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt', 'r')



#for i in dd:

    #print(i)

dd.close()
train 
test 
y = train['SalePrice']
train_x = train.drop('SalePrice', axis=1)
train_x
traintest = pd.concat([train_x,test]).reset_index()
traintest
traintest.info()
traintest.select_dtypes('object')  
traintest.select_dtypes('int64')
traintest_int64 = traintest.select_dtypes('int64').fillna(0)
traintest.select_dtypes('float64')
traintest_float64 = traintest.select_dtypes('float64').fillna(0)
oenc = OrdinalEncoder()

traintest_objenc = pd.DataFrame(oenc.fit_transform(traintest.select_dtypes('object').fillna('0')),columns = traintest.select_dtypes('object').columns)
traintest_objenc
traintest_nona_enc = traintest_int64.join(traintest_objenc).join(traintest_float64)
traintest_nona_enc
traintest_nona_enc.info()
traintest_norm =  traintest_nona_enc[['index','Id']].join(pd.DataFrame(StandardScaler().fit_transform(traintest_nona_enc.drop(['index','Id'], axis=1)), columns = traintest_nona_enc.drop(['index','Id'], axis=1).columns))

#traintest_norm =  traintest_nona_enc[['index','Id']].join(pd.DataFrame(MinMaxScaler().fit_transform(traintest_nona_enc.drop(['index','Id'], axis=1)), columns = traintest_nona_enc.drop(['index','Id'], axis=1).columns))
traintest_norm
train_norm = traintest_norm[0:1460].drop('index', axis=1)
train_norm
test_norm = traintest_norm[1460::].drop('index', axis=1).reset_index().drop('index',axis=1)
test_norm




X_train, X_test, y_train, y_test = train_test_split(train_norm, y)



modelgbr = GradientBoostingRegressor(n_estimators = 200, criterion='mae', random_state=42)

modelgbr.fit(X_train, y_train)

y_predgbr = modelgbr.predict(X_test)

mean_absolute_error(y_test, y_predgbr)
pd.DataFrame(modelgbr.feature_importances_, X_train.columns).sort_values(by=0,ascending=False).head(20)
#modelclf = GradientBoostingRegressor(criterion='mae', random_state=42)

#parameters = {'n_estimators':[50,200,300], 'learning_rate':[0.001,0.01,0.1],'max_depth':[1,3,6]}

#clf = GridSearchCV(modelclf, parameters)

#clf.fit(X_train, y_train)

#y_predclf = clf.predict(X_test)

#mean_absolute_error(y_test, y_predclf)

#clf.cv_results_

#clf.best_params_

#{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}
test_id = test_norm['Id']
modelsub = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 3, n_estimators = 200, criterion='mae', random_state=42)

modelsub.fit(train_norm.drop('Id',axis=1), y)

preds_test = modelsub.predict(test_norm.drop('Id',axis=1))





output = pd.DataFrame({'Id': test_id.values,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)


