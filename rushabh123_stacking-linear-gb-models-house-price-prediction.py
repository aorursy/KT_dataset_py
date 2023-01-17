# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer,LabelEncoder,StandardScaler,PolynomialFeatures,OneHotEncoder,RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from xgboost import XGBRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import skew,boxcox
import math
from lightgbm import LGBMRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
data = data.drop('Id',axis=1)
X_test = pd.read_csv('../input/test.csv')
X_test = X_test.drop('Id',axis=1)
X_train = data.drop('SalePrice',axis=1)
y_train = data['SalePrice']
X_combined = X_train.append(X_test,ignore_index=True)
columns_grouped_by_datatypes = X_combined.columns.to_series().groupby(X_combined.dtypes).groups
columns_grouped_by_datatypes = {k.name: v for k, v in columns_grouped_by_datatypes.items()}
continuous_variables = list(columns_grouped_by_datatypes['int64'].append(columns_grouped_by_datatypes['float64']))
categorical_variables = list(columns_grouped_by_datatypes['object'])
continuous_variables.remove('MSSubClass')
continuous_variables.remove('OverallQual')
continuous_variables.remove('OverallCond')
continuous_variables.remove('MoSold')
categorical_variables.append('MSSubClass')
categorical_variables.append('OverallQual')
categorical_variables.append('OverallCond')
categorical_variables.append('MoSold')
X_combined['MSZoning'] = X_combined['MSZoning'].fillna('Others')
X_combined['Alley'] = X_combined['Alley'].fillna('NoAccess')
X_combined['Utilities'] = X_combined['Utilities'].fillna('None')
X_combined['Exterior1st'] = X_combined['Exterior1st'].fillna('Other')
X_combined['Exterior2nd'] = X_combined['Exterior2nd'].fillna('Other')
X_combined['MasVnrType'] = X_combined['MasVnrType'].fillna('None')
X_combined['BsmtQual'] = X_combined['BsmtQual'].fillna('NoBasement')
X_combined['BsmtCond'] = X_combined['BsmtCond'].fillna('NoBasement')
X_combined['BsmtExposure'] = X_combined['BsmtExposure'].fillna('NoBasement')
X_combined['BsmtFinType1'] = X_combined['BsmtFinType1'].fillna('NoBasement')
X_combined['BsmtFinType2'] = X_combined['BsmtFinType2'].fillna('NoBasement')
X_combined['KitchenQual'] = X_combined['KitchenQual'].fillna('Po')
X_combined['Electrical'] = X_combined['Electrical'].fillna('SBrkr')
X_combined['Functional'] = X_combined['Functional'].fillna('Typ')
X_combined['FireplaceQu'] = X_combined['FireplaceQu'].fillna('NoFirePlace')
X_combined['GarageType'] = X_combined['GarageType'].fillna('NoGarage')
X_combined['GarageFinish'] = X_combined['GarageFinish'].fillna('NoGarage')
X_combined['GarageQual'] = X_combined['GarageQual'].fillna('NoGarage')
X_combined['GarageCond'] = X_combined['GarageCond'].fillna('NoGarage')
X_combined['PoolQC'] = X_combined['PoolQC'].fillna('NoPool')
X_combined['Fence'] = X_combined['Fence'].fillna('NoFence')
X_combined['MiscFeature'] = X_combined['MiscFeature'].fillna('None')
X_combined['SaleType'] = X_combined['SaleType'].fillna('Oth')
X_combined['MasVnrArea'] = X_combined['MasVnrArea'].fillna(0)
X_combined['GarageYrBlt'] = X_combined['GarageYrBlt'].fillna(0)
X_combined['GarageYrBlt'] = X_combined['GarageYrBlt'].fillna(0)
X_combined['BsmtFullBath'] = X_combined['BsmtFullBath'].fillna(0)
X_combined['BsmtHalfBath'] = X_combined['BsmtHalfBath'].fillna(0)
X_combined['BsmtFinSF1'] = X_combined['BsmtFinSF1'].fillna(0)
X_combined['BsmtFinSF2'] = X_combined['BsmtFinSF2'].fillna(0)
X_combined['GarageCars'] = X_combined['GarageCars'].fillna(0)
X_combined['GarageArea'] = X_combined['GarageArea'].fillna(0)
X_combined['BsmtUnfSF'] = X_combined['BsmtUnfSF'].fillna(0)
X_combined['TotalBsmtSF'] = X_combined['TotalBsmtSF'].fillna(0)
X_combined['LotFrontage'] = X_combined.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
X_combined['YearBuilt'] = 2019-X_combined['YearBuilt']
X_combined['YearRemodAdd'] = 2019-X_combined['YearRemodAdd']
X_combined['YrSold'] = 2019-X_combined['YrSold']
X_combined['GarageYrBlt'] = 2019-X_combined['GarageYrBlt']
X_combined['MSSubClass'] = X_combined['MSSubClass'].astype(str)
X_combined['OverallCond'] = X_combined['OverallCond'].astype(str)
X_combined['OverallQual'] = X_combined['OverallQual'].astype(str)
X_combined['MoSold'] = X_combined['MoSold'].astype(str)
corr_df = X_combined[continuous_variables]
corr_heatmap = corr_df.corr()
plt.figure(figsize=(17,17))
sns.heatmap(corr_heatmap,xticklabels=corr_df.columns,yticklabels=corr_df.columns)
continuous_variables_to_be_dropped = ['GarageCars','GarageYrBlt','TotRmsAbvGrd','LotFrontage','BsmtUnfSF','EnclosedPorch','TotalBsmtSF','BsmtFinSF2']
X_combined = X_combined.drop(continuous_variables_to_be_dropped,axis=1)
for column in continuous_variables_to_be_dropped:
    continuous_variables.remove(column)
y_train = np.log1p(y_train)
X_combined = pd.get_dummies(X_combined)
skewness = pd.DataFrame(data={'name':continuous_variables,'skewness':skew(X_combined[continuous_variables])})
highly_skewed_variables = skewness[abs(skewness['skewness'])>=1]['name']
for variable in highly_skewed_variables:
    X_combined[variable] = np.log1p(X_combined[variable])
robust_scaler = RobustScaler()
for variable in continuous_variables:
    X_combined[variable] = robust_scaler.fit_transform(X=X_combined[[variable]])
X_train = X_combined.iloc[:1460,:]
X_test = X_combined.iloc[1460:,:]
model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X_train,y_train)
y_pred_xgb_regressor = np.expm1(model_xgb.predict(X_test))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(X_train,y_train)
y_pred_gb_regressor = pd.Series(np.expm1(GBoost.predict(X_test)))
enet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
enet.fit(X_train,y_train)
y_pred_enet_regressor = np.expm1(enet.predict(X_test))
lasso = Lasso(alpha=0.001)
lasso.fit(X_train,y_train)
y_pred_lasso_regressor = np.expm1(lasso.predict(X_test))
light_gradient_boost = LGBMRegressor(max_depth=5,min_data_in_leaf=5,feature_fraction=0.5,bagging_fraction=0.5)
light_gradient_boost.fit(X_train,y_train)
y_pred_lgb_regressor = np.expm1(light_gradient_boost.predict(X_test))
#y_pred_stacking_regressor = (y_pred_xgb_regressor+y_pred_gb_regressor+y_pred_enet_regressor+y_pred_lasso_regressor+y_pred_lgb_regressor)/5
y_pred_xgb_regressor_train = np.expm1(model_xgb.predict(X_train))
y_pred_gb_regressor_train = np.expm1(GBoost.predict(X_train))
y_pred_enet_regressor_train = np.expm1(enet.predict(X_train))
y_pred_lasso_regressor_train = np.expm1(lasso.predict(X_train))
y_pred_lgb_regressor_train = np.expm1(light_gradient_boost.predict(X_train))
X_stack=pd.DataFrame()
X_stack['xgb_pred']=y_pred_xgb_regressor_train
X_stack['gb_pred']=y_pred_gb_regressor_train
#X_stack['enet_pred']=y_pred_enet_regressor_train
#X_stack['lasso_pred']=y_pred_lasso_regressor_train
X_stack['xlgb_pred']=y_pred_lgb_regressor_train
X_stack['true']=np.expm1(y_train)
for variable in list(X_stack):
    print(np.sum(abs(X_stack['true']-X_stack[variable]))/1000)
linear_regression = LinearRegression(normalize=True)
linear_regression.fit(X_stack.drop('true',axis=1),X_stack['true'])
y_pred = linear_regression.predict(X_stack.drop('true',axis=1))
X_stack['predicted'] = y_pred
X1_stack=pd.DataFrame()
X1_stack['xgb_pred']=y_pred_xgb_regressor
X1_stack['gb_pred']=y_pred_gb_regressor
#X1_stack['enet_pred']=y_pred_enet_regressor
#X1_stack['lasso_pred']=y_pred_lasso_regressor
X1_stack['xlgb_pred']=y_pred_lgb_regressor
y_pred_stacking_regressor = linear_regression.predict(X1_stack)
result_stacking_regressor = pd.DataFrame()
result_stacking_regressor['Id'] = range(1461,2920)
result_stacking_regressor['SalePrice']=y_pred_gb_regressor
result_stacking_regressor.to_csv('./submission.csv',index=False)
