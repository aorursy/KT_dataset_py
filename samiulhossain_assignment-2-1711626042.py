# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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
import numpy as np
import pandas as pd 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head(5)
test.head(5)
test_ID = test['Id']
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
import missingno as msno
msno.bar(train,labels=True,fontsize=12)
msno.bar(test,labels=True,fontsize=12)
null_col =  ['Alley', 'FireplaceQu','PoolQC', 'Fence', 'MiscFeature']
train = train.drop(columns=null_col)
test = test.drop(columns=null_col)
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
!pip install fancyimpute
obj_list = train.select_dtypes('object').columns
print(obj_list)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col in obj_list:
    if col in train.columns:
        i = train.columns.get_loc(col)
        train.iloc[:,i] = train.apply(lambda i:labelencoder.fit_transform(i.astype(str)), axis=0, result_type='expand')

train.head()
from fancyimpute import KNN 
knn_imputer = KNN()
train_knn = train.copy(deep=True)
train_knn.iloc[:,:] = knn_imputer.fit_transform(train_knn)
train_knn.info()
obj_list = test.select_dtypes('object').columns
print(obj_list)
labelencoder = LabelEncoder()
for col in obj_list:
    if col in test.columns:
        i = test.columns.get_loc(col)
        test.iloc[:,i] = test.apply(lambda i:labelencoder.fit_transform(i.astype(str)), axis=0, result_type='expand')

test.head()
knn_imputer2 = KNN()
test_knn = test.copy(deep=True)
test_knn.iloc[:,:] = knn_imputer2.fit_transform(test_knn)
test_knn.info()
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import sklearn.metrics as mtc
x = train_knn.drop('SalePrice',axis=1).values
y = train_knn['SalePrice'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
rdg = Ridge()
rdg.fit(x_train,y_train)
y_pred=rdg.predict(x_test)
print("Test R2 Score:",mtc.r2_score(y_test,y_pred))
lss = Lasso()
lss.fit(x_train,y_train)
y_pred=lss.predict(x_test)
print("Test R2 Score:",mtc.r2_score(y_test,y_pred))
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)
y_pred=rfr.predict(x_test)
print("Test R2 Score:",mtc.r2_score(y_test,y_pred))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
gbr.fit(x_train,y_train)
y_pred=gbr.predict(x_test)
print("Test R2 Score:",mtc.r2_score(y_test,y_pred))
y_pred.shape
predictions = gbr.predict(test_knn)
predictions
submission = pd.DataFrame({'Id':test_ID,'SalePrice':predictions})
submission.head()
filename = 'demo_pred4.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
