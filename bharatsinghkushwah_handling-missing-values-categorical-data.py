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
#creating dataframes of testing and trainig data
train= pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test= pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#let us see how the training and testing data look
train.head()
test.head()
#splitting the data into training and validation sets
from sklearn.model_selection import train_test_split
X=train.drop("SalePrice",axis=1)
y=train.SalePrice
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1,
                                                      random_state=0)
y.isnull().sum()
print(X_train.shape) 
print(X_valid.shape)
B=[]
for col in X_train.columns:
    if(X_train[col].isnull().sum()>10):
        B.append(col)
X_train[B].isnull().sum()
X_train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
X_valid.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
#let us see the categorical columns
cat_col=[]
for col in X_train.columns:
    if(X_train[col].dtype==object):
        cat_col.append(col)
print(cat_col)
num_col=[]
num_col=list(X_train.drop(cat_col,axis=1).columns)
#we dont want taarget variable in feature sets
train[cat_col].nunique(axis=0,dropna=True)
import category_encoders as ce
tar_encoder= ce.TargetEncoder(cols= cat_col)
tar_encoder.fit(X_train[cat_col],y_train)
X_train= X_train.join(tar_encoder.transform(X_train[cat_col]).add_suffix("_te"))
X_valid= X_valid.join(tar_encoder.transform(X_valid[cat_col]).add_suffix("_te"))
X_train.head()
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='median')
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train.drop(cat_col,axis=1)))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid.drop(cat_col,axis=1)))
imputed_X_train.columns=X_train.drop(cat_col,axis=1).columns
imputed_X_valid.columns=X_valid.drop(cat_col,axis=1).columns
# let us create a xgbclassifier model to fit the data and validate it
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
reg=XGBRegressor(max_depth= 6, learning_rate=0.05,n_estimators=5000, n_jobs=-1,early_stopping_rounds=40)
reg.fit(imputed_X_train,y_train)
pred=reg.predict(imputed_X_valid)
print(mae(pred,y_valid))
test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
# let us see our accuracy on submission
# first we need to target encode and remove missing values from test data
X_test= test.join(tar_encoder.transform(test[cat_col]).add_suffix("_te")).drop(cat_col,axis=1)
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))
imputed_X_test.columns=X_test.columns
X_test['SalePrice']=reg.predict(imputed_X_test)
X_test[['Id','SalePrice']].to_csv("submission.csv",index=False)
X_train[cat_col].nunique()
test[cat_col].nunique()
OH_col=cat_col.copy()
OH_col.remove('Neighborhood')
OH_col.remove('Exterior1st')
OH_col.remove('Exterior2nd')
OH_col
imp_X_train=pd.DataFrame()
imp_X_valid=pd.DataFrame()

my_imputer = SimpleImputer(strategy='median')
imp_X_train[num_col]=pd.DataFrame(my_imputer.fit_transform(X_train[num_col]))
imp_X_valid[num_col]=pd.DataFrame(my_imputer.transform(X_valid[num_col]))

my_imputer = SimpleImputer(strategy='most_frequent')
imp_X_train[cat_col]=pd.DataFrame(my_imputer.fit_transform(X_train[cat_col]))
imp_X_valid[cat_col]=pd.DataFrame(my_imputer.transform(X_valid[cat_col]))

imp_X_train.head()
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_train = pd.DataFrame(OH_encoder.fit_transform(imp_X_train[OH_col]))
OH_valid = pd.DataFrame(OH_encoder.transform(imp_X_valid[OH_col]))

# One-hot encoding removed index, we need to put it back
OH_train.index = X_train.index
OH_valid.index = X_valid.index
#we need to add the numeric columns
OH_train=OH_train.join(imp_X_train[num_col])
OH_valid=OH_valid.join(imp_X_valid[num_col])

#we need add the three categorical cols we left as target encoded
L=['Neighborhood','Exterior1st','Exterior2nd']
tar_encoder= ce.TargetEncoder(cols= L)
tar_encoder.fit(imp_X_train[L],y_train)
OH_train= OH_train.join(tar_encoder.transform(imp_X_train[L],y_train))
OH_valid= OH_valid.join(tar_encoder.transform(imp_X_valid[L]))
OH_train.shape
reg=XGBRegressor(max_depth= 30, learning_rate=0.05,n_estimators=5000, n_jobs=-1,early_stopping_rounds=40)
reg.fit(OH_train,y_train)
pred=reg.predict(OH_valid)
print(mae(pred,y_valid))
