# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")
pd.set_option('Display.max_columns',500)
pd.set_option('max_rows',500)
# Any results you write to the current directory are saved as output.
train_data.head(2)
test_data.head(2)
train_data.describe()
train_data['SalePrice'].describe()
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,8))
plt.plot(train_data['SalePrice'])
plt.show()
target = train_data['SalePrice']
target_log = np.log1p(train_data['SalePrice'])
target_log.head(5)
# drop target variable from train dataset
train = train_data.drop(["SalePrice"], axis=1)
train.columns
# Concante the train and test data
train_objs_num=len(train)
data=pd.concat(objs=[train,test_data],axis=0)
data.shape
# save all categorical columns in list
categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']
# dataframe with categorical features
data_cat = data[categorical_columns]
# dataframe with numerical features
data_num = data.drop(categorical_columns, axis=1)
data_cat.head(2)
data_num.head(2)

# Sum Features has more than 80% null value so we drop these column
data_cat.drop(['Alley','MiscFeature'],axis=1,inplace=True)
data_cat.drop('FireplaceQu',axis=1,inplace=True)
data_cat.drop(['PoolQC','Fence'],axis=1,inplace=True)
data_cat.isnull().sum()/len(train_data)*100
#  Now fill the missing value
data_cat.fillna(method='ffill',inplace=True)
data_num.isnull().sum()/len(test_data)
data_num.fillna(method='ffill',inplace=True)
data_cat.shape
data_num.shape

data_cat_dummy = pd.get_dummies(data_cat)
data = pd.concat([data_num, data_cat_dummy], axis=1)
train_data1=(data[:train_objs_num])
test_data1 = (data[train_objs_num:])
train_data.shape
train_data1.head(2)
train_data1['SalePrice']=train_data['SalePrice']
test_data1.shape
train_data1.shape
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
train,test = train_test_split(train_data1,test_size = .33,random_state=100)
train_y = train['SalePrice']
test_y = test['SalePrice']

train_x  = train.drop('SalePrice',axis=1)
test_x = test.drop('SalePrice',axis=1)
lm = LinearRegression()
lm.fit(train_x,train_y)

lm_pred = lm.predict(test_x)
lm_pred[:5]
df_pred = pd.DataFrame({'Actual':test_y,'Pred':lm_pred})
df_pred.head(4)

mse = mean_squared_error(test_y,lm_pred)
mse
rmse = np.sqrt(mse)
rmse
r2_sc = r2_score(test_y,lm_pred)
r2_sc
from sklearn.ensemble import RandomForestRegressor
rf_lm = RandomForestRegressor()
rf_lm.fit(train_x,train_y)

rf_pred = rf_lm.predict(test_x)
rf_pred[:5]
rf_pred_model = pd.DataFrame({'Actual':test_y,'Pred':rf_pred})
rf_pred_model.head(5)
mse_rf = mean_squared_error(test_y,rf_pred)

rmse_rf = np.sqrt(mse)
print(mse_rf,rmse_rf)

import xgboost 
from xgboost import XGBRegressor

model_xg = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
model_xg.fit(train_x,train_y)
xg_pred = model_xg.predict(test_x)
xg_pred[:5]

xg_mse = mean_squared_error(test_y,xg_pred)
xg_mse
rmse_xg = np.sqrt(mse)
rmse_xg
r2_xg = r2_score(test_y,xg_pred)
r2_xg
from sklearn.metrics import explained_variance_score
explained_lm_sc = explained_variance_score(test_y,xg_pred)
explained_lm_sc
model_xg_final = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
model_xg_final.fit(train_x,train_y)
model_final = model_xg_final.predict(test_data1)
model_final[:5]
df = pd.DataFrame({'SalePrice':model_final})
df['Id'] = test_data1.index+1
df.head()
df1 = df[['Id','SalePrice']]

df1.head()
df1[['Id','SalePrice']].to_csv('Submission.csv',index=False)
