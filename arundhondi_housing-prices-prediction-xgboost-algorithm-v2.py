# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import statements
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#reAD TRAIN AND TEST DATA
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
test_id=test_data.Id
print(train_data.columns)
#train and test data removing oject related data type and removing null values
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
#1.droping catogorial values 
# y=train_data.SalePrice
# x=train_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
# test_dat=test_data.select_dtypes(exclude=['object'])
print(train_data.shape)

#2. one hot encoding matrix for catogorial values.
# y=train_data.SalePrice
# x_train=train_data.drop(['SalePrice'], axis=1)
# final_x=pd.get_dummies(x_train)
# final_test_dat=pd.get_dummies(test_data)
# x,test_dat=final_x.align(final_test_dat,join='left',axis=1)
# test_dat.head()


#3. selecting features that are neccessary
cols= [ 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
        'LotShape', 'LandContour',  'LotConfig',
       'LandSlope',  'Condition1',  'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond',  
       'RoofStyle',  'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual',   'BsmtQual',
       'BsmtCond',  'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       '1stFlrSF', '2ndFlrSF',
       'GrLivArea', 'BedroomAbvGr', 
       'TotRmsAbvGrd', 'GarageArea',
       'GarageCond', 'WoodDeckSF', 'OpenPorchSF',
        'SaleType',
       'SaleCondition']
y=train_data.SalePrice
x_train=train_data.drop(['SalePrice'], axis=1)
x_train_col=x_train[cols]
test_data_col=test_data[cols]
final_x=pd.get_dummies(x_train_col)
final_test_dat=pd.get_dummies(test_data_col)
x_na,test_dat_na=final_x.align(final_test_dat,join='left',axis=1)
x=x_na.fillna(x_na.mean())
test_dat=test_dat_na.fillna(test_dat_na.mean())
test_dat.shape
#assigning mean values
my_imputer = SimpleImputer()
X= my_imputer.fit_transform(x)
test_data=my_imputer.transform(test_dat)

print(X.shape)
print((test_data.shape))
#splitting the data into test and train data sets for fitting the model
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25,random_state=1 )
#model 
my_model = XGBRegressor(n_estimators=1000,learning_rate=0.0275)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
# make predictions
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
predictions_train = my_model.predict(train_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions_train, train_y)))
#test data of the model
predicted_prices = my_model.predict(test_data)

#submission
my_submission = pd.DataFrame({'Id': test_id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('hp_XG_submission.csv', index=False)
