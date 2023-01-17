

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.sample(5)
test.sample(5)
from catboost import CatBoostRegressor

y=train.SalePrice
cat_cols=train.select_dtypes(include='object').columns
cat_cols
train
train=train.drop(['Id'],axis=1)
train=train.drop(['SalePrice'],axis=1)
train.columns
train=train.fillna(0)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,y,random_state=22)
cat_regressor=CatBoostRegressor()
cat_regressor.fit(x_train,y_train,cat_features=cat_cols)
ids=test['Id']
test=test.drop(['Id'],axis=1)


test=test.fillna(0)
predicted=cat_regressor.predict(test)
sub=pd.DataFrame({'Id':ids,'SalePrice':predicted})
sub.to_csv('submission.csv')
sub_type=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
sub_type