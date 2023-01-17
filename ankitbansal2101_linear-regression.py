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
dataset_train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
dataset_train.head()

dataset_test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
dataset_test.head()
import matplotlib.pyplot as plt

plt.hist(dataset_train.SalePrice)
# histogram for saleprice is skewed so we took log of saleprice
dataset_train.SalePrice=np.log(dataset_train.SalePrice)

dataset_train.SalePrice
plt.hist(dataset_train.SalePrice)
dataset_train.isnull().sum()
dataset_train.dropna(axis=1,inplace=True)
dataset_train.shape
obj_df = dataset_train.select_dtypes(include=['object']).copy()

obj_df.columns

dataset_train=pd.get_dummies(dataset_train,columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',
       'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive',
       'SaleType', 'SaleCondition'],drop_first=True)
dataset_train.shape
dataset_train.dropna(axis=1,inplace=True)
dataset_train.head()
dataset_train.columns
# drop irrelvant features
dataset_train.drop("Id",axis=1,inplace=True)

# drop na values
dataset_test.dropna(axis=1,inplace=True)
obj_df = dataset_test.select_dtypes(include=['object']).copy()

obj_df.columns
# get dummies
dataset_test.head()
dataset_test=pd.get_dummies(dataset_test,columns=['Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'PavedDrive',
       'SaleCondition'],drop_first=True)
dataset_test.shape
dataset_test.shape
dataset_test.isnull().sum()
submission = pd.DataFrame()
submission['Id'] = dataset_test.Id
testCols = dataset_test.columns
dataset_test.drop("Id",axis=1,inplace=True)
dataset_test = dataset_test[dataset_train.columns & testCols]
dataset_test

submission['Id']
# create feature matrix and target array
X = dataset_train[dataset_train.columns & dataset_test.columns]
y=dataset_train["SalePrice"]

y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LinearRegression
model_reg=LinearRegression()
model_reg.fit(X_train,y_train)
# predictions on test data
y_pred=model_reg.predict(X_test)
y_pred
# r-squared for test data
sse=np.sum((y_test-y_pred)**2)
sst=np.sum((np.mean(y_train)-y_test)**2)
r2=1-sse/sst
r2
#rmse for test data
rmse=np.sqrt(sse/X_test.shape[0])
rmse
# predictions on given test dataset
predictions=model_reg.predict(dataset_test)
final_predictions=np.exp(predictions)
final_predictions
from sklearn.linear_model import Ridge
model_ridge=Ridge(alpha=6)
model_ridge.fit(X_train,y_train)
# predictions on test data
y_pred_ridge=model_ridge.predict(X_test)

# r-squared for test data
sse=np.sum((y_test-y_pred_ridge)**2)
sst=np.sum((np.mean(y_train)-y_test)**2)
r2=1-sse/sst
r2
from sklearn.linear_model import Lasso
model_lasso=Lasso()
model_lasso.fit(X_train,y_train)
# predictions on test data
y_pred_lasso=model_lasso.predict(X_test)

# r-squared for test data
sse=np.sum((y_test-y_pred_lasso)**2)
sst=np.sum((np.mean(y_train)-y_test)**2)
r2=1-sse/sst
r2
from sklearn.tree import DecisionTreeRegressor
model_tree=DecisionTreeRegressor(max_depth=3)
model_tree.fit(X_train,y_train)
# predictions on test data
y_pred_tree=model_tree.predict(X_test)
y_pred_tree
# r-squared for test data
sse_tree=np.sum((y_test-y_pred_tree)**2)
sst_tree=np.sum((np.mean(y_train)-y_test)**2)
r2_tree=1-sse_tree/sst_tree
r2_tree
from sklearn.ensemble import RandomForestRegressor
model_rf=RandomForestRegressor()
model_rf.fit(X_train,y_train)
y_pred_rf=model_rf.predict(X_test)
# r-squared for test data
sse_rf=np.sum((y_test-y_pred_rf)**2)
sst_rf=np.sum((np.mean(y_train)-y_test)**2)
r2_rf=1-sse_rf/sst_rf
r2_rf

model_rf.feature_importances_ 
data = pd.Series(data=model_rf.feature_importances_,index=X_train.columns)
data.sort_values(ascending=True,inplace=True)
plt.figure(figsize=(15,15))
data.plot.barh()
from sklearn.model_selection import GridSearchCV
parameters={"max_depth":np.arange(1,7),"max_features":np.arange(10,100)}
tune_model = GridSearchCV(model_rf,parameters,cv=5)
tune_model.fit(X_train,y_train)
tune_model.best_params_
y_pred_rf_tune=tune_model.predict(X_test)
# r-squared for test data
sse_rf_tune=np.sum((y_test-y_pred_rf_tune)**2)
sst_rf_tune=np.sum((np.mean(y_train)-y_test)**2)
r2_rf_tune=1-sse_rf_tune/sst_rf_tune
r2_rf_tune
# predictions on given test dataset
predictions_ridge=model_ridge.predict(dataset_test)
final_predictions=np.exp(predictions_ridge)
final_predictions
submission["SalePrice"]=final_predictions
submission
submission.to_csv("submission_regression1.csv",index=False)
