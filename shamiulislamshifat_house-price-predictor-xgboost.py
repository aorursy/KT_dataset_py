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
#import lib

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from tqdm import tqdm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
# load datasets

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.columns = train.columns

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

print('Train:{}   Test:{}'.format(train.shape,test.shape))
train.head(5)
#delete missing values from training and test datasets


missing = test.isnull().sum()
missing = missing[missing>0]
train.drop(missing.index, axis=1, inplace=True)
train.drop(['Electrical'], axis=1, inplace=True)

test.dropna(axis=1, inplace=True)
test.drop(['Electrical'], axis=1, inplace=True)
#using matching technique

l_test = tqdm(range(0, len(test)), desc='Matching')
for i in l_test:
    for j in range(0, len(train)):
        for k in range(1, len(test.columns)):
            if test.iloc[i,k] == train.iloc[j,k]:
                continue
            else:
                break
        else:
            submission.iloc[i, 1] = train.iloc[j, -1]
            break
l_test.close()
#save submission to csv

#submission.to_csv('submission.csv', index=False)
# import library
import seaborn as sns

import os

import warnings
warnings.filterwarnings("ignore")
#explore train data

ft_pred = list(set(train.columns) - set(test.columns))
train[ft_pred].describe()

#Plot GrLivArea vs SalePrice
plt.scatter(train['GrLivArea'], train['SalePrice'], color='blue', alpha=0.5)
plt.title("LotArea vs SalePrice")
plt.legend(loc='best')
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

#Importing packages


import matplotlib.pyplot as plt
import seaborn as sns

import os

import warnings
warnings.filterwarnings("ignore")
#Feature to predict
ft_pred = list(set(train.columns) - set(test.columns))
ft_pred
#FILL NAN VALUES 

#save and drop id
train_id = train["Id"]
train.drop(columns='Id',inplace=True)

test_id = test["Id"]
test.drop(columns='Id',inplace=True)

#select object columns
obj_col = train.columns[train.dtypes == 'object'].values

#select non object columns
num_col = train.columns[train.dtypes != 'object'].values
num_col_test = test.columns[test.dtypes != 'object'].values

#replace null value in obj columns with None
train[obj_col] = train[obj_col].fillna('None')
test[obj_col] = test[obj_col].fillna('None')

#replace null value in numeric columns with 0
train[num_col] = train[num_col].fillna(0)
test[num_col_test] = test[num_col_test].fillna(0)

train_001 = train
test_001 = test
#one hot encoding

import category_encoders as ce

#Ordinal features
ordinal_features = ["ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure", "BsmtFinType1","BsmtFinType2",
                    "HeatingQC","Electrical","KitchenQual", "FireplaceQu","GarageQual","GarageCond","PoolQC"]

#Split X,y
train_002_X = train_001.drop(ft_pred, axis=1)
train_002_y = train_001[ft_pred]

ce_one_hot = ce.OrdinalEncoder(cols = ordinal_features)

train_003 = pd.concat([ce_one_hot.fit_transform(train_002_X), train_002_y], axis=1, sort=False)
test_003  = ce_one_hot.transform(test_001)

#Nominal features
nominal_features = [x for x in obj_col if x not in ordinal_features]

#Transfer object to int
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#for loop nominal feature column
for i in train_003[nominal_features].columns:
    #fit and transform each column and assign to itself
    train_003[i] = labelencoder.fit_transform(train_003[i])
    
#for loop nominal feature column
for i in test_003[nominal_features].columns:
    #fit and transform each column and assign to itself
    test_003[i] = labelencoder.fit_transform(test_003[i])
    
#Get dummy variable for nominal features
train_005 = pd.get_dummies(train_003,columns=nominal_features,drop_first=True)
test_005 = pd.get_dummies(test_003,columns=nominal_features,drop_first=True)
#Only for test set
#Check if any null values
print(train_005.isnull().any().sum())
print(test_005.isnull().any().sum())

#Get missing columns in the training test
missing_cols = set(train_005.drop(columns="SalePrice").columns) - set(test_005.columns)

#Add a missing column in test set with default value equal to 0
for cols in missing_cols:
    test_005[cols] = 0
    
#Ensure the order of column in the test set is in the same order than in train set
test_005 = test_005[train_005.drop(columns="SalePrice").columns]
#select best features



#TotalBath
train_005['TotalBath'] = (train_005['FullBath'] + train_005['HalfBath'] + train_005['BsmtFullBath'] + train_005['BsmtHalfBath'])
test_005['TotalBath']  = (test_005['FullBath']  + test_005['HalfBath']  + test_005['BsmtFullBath']  + test_005['BsmtHalfBath'])

#TotalPorch
train_005['TotalPorch'] = (train_005['OpenPorchSF'] + train_005['3SsnPorch'] + train_005['EnclosedPorch'] + train_005['ScreenPorch'] + train_005['WoodDeckSF'])
test_005['TotalPorch']  = (test_005['OpenPorchSF']  + test_005['3SsnPorch']  + test_005['EnclosedPorch']  + test_005['ScreenPorch']    + test_005['WoodDeckSF'])

#Modeling happen during the sale year
train_005["RecentRemodel"] = (train_005["YearRemodAdd"] == train_005["YrSold"]) * 1
test_005["RecentRemodel"]  = (test_005["YearRemodAdd"]  == test_005["YrSold"]) * 1

#House sold in the year it was built
train_005["NewHouse"] = (train_005["YearBuilt"] == train_005["YrSold"]) * 1
test_005["NewHouse"]  = (test_005["YearBuilt"]  == test_005["YrSold"]) * 1

#HasPool
train_005['HasPool'] = train_005['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test_005['HasPool']  = test_005['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

#HasFireplaces
train_005['HasFirePlace'] = train_005['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
test_005['HasFirePlace']  = test_005['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#Has2ndFloor
train_005['Has2ndFloor'] = train_005['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test_005['Has2ndFloor']  = test_005['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

#HasGarage
train_005['HasGarage'] = train_005['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
test_005['HasGarage']  = test_005['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

#HasBsmnt
train_005['HasBsmnt'] = train_005['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test_005['HasBsmnt']  = test_005['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
#aplit data

#Importing packages
from sklearn.model_selection import train_test_split

X = train_005.drop(columns="SalePrice")
y = train_005["SalePrice"]

#Particiona o data set originalmente Train em Train(Treino) e Val(validação)
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state=0)
#use XGBOOST



#Importing Packages
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from sklearn.impute import SimpleImputer
#XGBoost hyper-parameter tuning
def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_
xgb_model = XGBRegressor(
        objective = 'reg:squarederror',
        colsample_bytree = 0.5,
        learning_rate = 0.05,
        max_depth = 6,
        min_child_weight = 1,
        n_estimators = 1000,
        subsample = 0.7)

%time xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_val, y_val)], verbose=False)

y_pred_xgb = xgb_model.predict(X_val)

#mae_xgb = mean_absolute_error(y_val, y_pred_xgb)

#print("MAE: ", mae_xgb)
#Plot Real vs Predict
plt.scatter(X_val['GrLivArea'] * 0.092903, y_val,          color='blue', label='Real',    alpha=0.5)
plt.scatter(X_val['GrLivArea'] * 0.092903, y_pred_xgb,  color='red' , label='Predict', alpha=0.5)
plt.title("Real vs Predict")
plt.legend(loc='best')
plt.show()
#predict

X_test = test_005

# Use the model to make predictions
y_pred_test = xgb_model.predict(X_test)

#submission2 = pd.DataFrame({'Id':test_id,'SalePrice':y_pred_test})

# Save results
#submission2.to_csv("submission1.csv",index=False)
lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=12000, 
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.4, 
                                       )
lgbm.fit(X_train, y_train,eval_metric='rmse')
predict6 = lgbm.predict(X_val)
print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_val, predict6))))
lgbm.fit(X, y,eval_metric='rmse')
predict7 = lgbm.predict(test_005)
submission3 = pd.DataFrame({'Id':test_id,'SalePrice':predict4})

# Save results
submission3.to_csv("submission3.csv",index=False)
predicty=y_pred_test*0.45+predict7*0.45

submission7 = pd.DataFrame({'Id':test_id,'SalePrice':predicty})

# Save results
submission6.to_csv("submission7.csv",index=False)




