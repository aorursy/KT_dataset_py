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
raw_data=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
raw_data.head()           
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
all_features=list((raw_data.columns))[1:(len(raw_data.columns)-1)]
train_set,validation_set=train_test_split(raw_data,train_size=0.8,test_size=0.2,random_state=42)
all_corr=train_set.corr(method='spearman').loc['SalePrice']
all_corr
select_features_cor={}
for pred,cor in list(all_corr.items()):
    abs_cor=abs(cor)
    if abs_cor>0.3:
        select_features_cor[pred]=cor
select_features=list(select_features_cor.keys())
select_features
cols_with_missing = [col for col in select_features
                     if train_set[col].isnull().any()]
cols_with_missing
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='median')
X = pd.get_dummies(train_set[select_features])
X_val = pd.get_dummies(validation_set[select_features])
imputed_train = pd.DataFrame(my_imputer.fit_transform(X))
imputed_valid = pd.DataFrame(my_imputer.transform(X_val))

# Imputation removed column names; put them back
imputed_train.columns = X.columns
imputed_valid.columns = X_val.columns
out_features=select_features.copy()
out_features.remove('SalePrice')
out_features
my_model = XGBRegressor(n_estimators=200, learning_rate=0.1,early_stopping_rounds=5)
my_model.fit(imputed_train[out_features],imputed_train['SalePrice'])
pred=my_model.predict(imputed_valid[out_features])
mean_absolute_error(pred, imputed_valid['SalePrice'])
test_data=pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
test_data.head()
test_pre=my_model.predict(test_data[out_features])
output_data=pd.DataFrame({'Id':test_data.Id,'SalePrice':test_pre})
output_data.to_csv('my_submission.csv',index=False)
