# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Analyze skewness
from scipy.stats import skew
# LabelEncoder
from sklearn.preprocessing import LabelEncoder
# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor

# Metrics for root mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print('train shap' , train.shape )
print('test shap' , test.shape )
print('train info' , train.info )
# Data Adjusting 
# Outliers detecting and Removing
train = train.drop(
    train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# Data Concatenation
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

# Drop utilities column
all_data = all_data.drop(['Utilities'], axis=1)
na_totals = all_data.isnull().sum().sort_values(ascending=False)
na_totals[na_totals>0]
# Impute missing data
# Impute missing categorical values
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'):
     all_data[col] = all_data[col].fillna('None') 
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType','MSSubClass'):
    all_data[col] = all_data[col].fillna('None')    
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

all_data["Functional"] = all_data["Functional"].fillna("Typ") # means typical

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
# Transform numerical to categorical
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
#Include "TotalSF" feature
# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# Transform skewed numerical data
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])
#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
# Encode and extract dummies from categorical features

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
all_data = pd.get_dummies(all_data)
print(train.shape[0])
# Select features
X_train = all_data[:train.shape[0]] # up to 1457 rows
X_test = all_data[train.shape[0]:] # strat from 1458 to the end

y = train.SalePrice
# Ensemble
# XGBoost + Lasso + ElasticNet
# Initialize models
lr = LinearRegression(n_jobs = -1)
rd = Ridge(alpha = 4.84)
rf = RandomForestRegressor( n_estimators = 12, max_depth = 3, n_jobs = -1)
gb = GradientBoostingRegressor( n_estimators = 40,max_depth = 2)
nn = MLPRegressor( hidden_layer_sizes = (90, 90), alpha = 2.75)
# Initialize Ensemble
model = StackingRegressor(regressors=[rf, gb, nn, rd],  meta_regressor=lr)
# Fit the model on our data
model.fit(X_train, y)
# Predict training set
y_pred = model.predict(X_train)
print(sqrt(mean_squared_error(y, y_pred)))
# Predict test set
Y_pred = model.predict(X_test)
# Submission
# Create empty submission dataframe
sub = pd.DataFrame()

# Insert ID and Predictions into dataframe
sub['Id'] = test['Id']
sub['SalePrice'] = np.expm1(Y_pred)

# Output submission file
sub.to_csv('submission1.csv',index=False)