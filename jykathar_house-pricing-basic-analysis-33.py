import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from xgboost import XGBRegressor
warnings.filterwarnings(action='ignore')                  # Turn off the warnings.
%matplotlib inline
# pandas 최대 출력 줄 수 설정 (결측치 column을 모두 보기 위해)
pd.set_option('display.max_rows', 221)
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print('train dataframe shape : {0}'.format(train_df.shape))
print('test dataframe shape : {0}'.format(test_df.shape))
train_df.head()
train_df.describe()
train_df.SalePrice.sort_values()
# The number of NA
train_df_na_num = train_df.isnull().sum()
train_df_na_num = train_df_na_num[train_df_na_num > 0]
train_df_na_num
# The ratio of NA
train_df_na_mean = train_df.isnull().mean()
train_df_na_mean = train_df_na_mean[train_df_na_mean > 0]
train_df_na_mean
test_df_na_num = test_df.isnull().sum()
test_df_na_num = test_df_na_num[test_df_na_num > 0]
test_df_na_num
test_df_na_mean = test_df.isnull().mean()
test_df_na_mean = test_df_na_mean[test_df_na_mean > 0]
test_df_na_mean
train_df.Utilities.value_counts()
plt.hist(train_df.Utilities)
plt.title('Utilities')
plt.show()
train_df = train_df.drop(['Id', 'Street', 'Alley', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities'], axis = 1)
test_df = test_df.drop(['Id', 'Street', 'Alley', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities'], axis = 1)
print('train dataframe shape : {0}'.format(train_df.shape))
print('test dataframe shape : {0}'.format(test_df.shape))
# The ratio of NA
train_df_na_mean = train_df.isnull().mean()
train_df_na_mean = train_df_na_mean[train_df_na_mean > 0]
train_df_na_mean
test_df_na_mean = test_df.isnull().mean()
test_df_na_mean = test_df_na_mean[test_df_na_mean > 0]
test_df_na_mean
model = LinearRegression()
X = train_df[['LotArea', '1stFlrSF']]
Y = train_df.LotFrontage
X_train = X[-Y.isnull()]
Y_train = Y[-Y.isnull()]
model.fit(X_train, Y_train)
Y_fill = model.predict(X)
train_df.LotFrontage[Y.isnull()] = np.round(Y_fill[Y.isnull()], 2)
model = LinearRegression()
X = test_df[['LotArea', '1stFlrSF']]
Y = test_df.LotFrontage
X_train = X[-Y.isnull()]
Y_train = Y[-Y.isnull()]
model.fit(X_train, Y_train)

Y_fill = model.predict(X)
test_df.LotFrontage[Y.isnull()] = np.round(Y_fill[Y.isnull()], 2)
train_df.MasVnrType[train_df.MasVnrType.isnull()] = 'None'
train_df.MasVnrArea[train_df.MasVnrArea.isnull()] = 0
test_df.MasVnrType[test_df.MasVnrType.isnull()] = 'None'
test_df.MasVnrArea[test_df.MasVnrArea.isnull()] = 0
train_df.Electrical[train_df.Electrical.isnull()] = 'SBrkr'
sns.scatterplot(x="GarageArea", y="SalePrice", hue = 'GarageType', data=train_df)
sns.boxplot(x = train_df.GarageType, y = train_df.SalePrice)
# The ratio of NA
train_df_na_mean = train_df.isnull().mean()
train_df_na_mean = train_df_na_mean[train_df_na_mean > 0]
train_df_na_mean
test_df_na_mean = test_df.isnull().mean()
test_df_na_mean = test_df_na_mean[test_df_na_mean > 0]
test_df_na_mean
train_df.BsmtQual.fillna('No', inplace = True)
train_df.BsmtCond.fillna('No', inplace = True)
train_df.BsmtExposure.fillna('No', inplace = True)
train_df.BsmtFinType1.fillna('No', inplace = True)
train_df.BsmtFinType2.fillna('No', inplace = True)

test_df.BsmtQual.fillna('No', inplace = True)
test_df.BsmtCond.fillna('No', inplace = True)
test_df.BsmtExposure.fillna('No', inplace = True)
test_df.BsmtFinType1.fillna('No', inplace = True)
test_df.BsmtFinType2.fillna('No', inplace = True)
test_df.BsmtFinSF1.fillna(0, inplace = True)
test_df.BsmtFinSF2.fillna(0, inplace = True)
test_df.BsmtUnfSF.fillna(0, inplace = True)
test_df.TotalBsmtSF.fillna(0, inplace = True)
train_df.GarageType.fillna('No', inplace = True)
train_df.GarageYrBlt.fillna('No', inplace = True)
train_df.GarageFinish.fillna('No', inplace = True)
train_df.GarageQual.fillna('No', inplace = True)
train_df.GarageCond.fillna('No', inplace = True)

test_df.GarageType.fillna('No', inplace = True)
test_df.GarageYrBlt.fillna('No', inplace = True)
test_df.GarageFinish.fillna('No', inplace = True)
test_df.GarageQual.fillna('No', inplace = True)
test_df.GarageCond.fillna('No', inplace = True)
test_df.GarageType[test_df.GarageCars.isnull()] = 'No'
test_df.GarageCars.fillna(0, inplace = True)
test_df.GarageArea.fillna(0, inplace = True)
train_df.FireplaceQu.fillna('No', inplace = True)
test_df.FireplaceQu.fillna('No', inplace = True)
# 최빈값인 RL로
test_df.MSZoning.fillna('RL', inplace = True)

#최빈값인 VinylSd로
test_df.Exterior1st.fillna('VinylSd', inplace = True)
test_df.Exterior2nd.fillna('VinylSd', inplace = True)
# Basement가 없는 주택의 욕실은 0으로 채움
test_df.BsmtFullBath.fillna(0, inplace = True)
test_df.BsmtHalfBath.fillna(0, inplace = True)

#최빈값 TA
test_df.KitchenQual.fillna('TA', inplace = True)

#최빈값 Typ
test_df.Functional.fillna('Typ', inplace = True)

#최빈값 WD
test_df.SaleType.fillna('WD', inplace = True)
len_train = len(train_df)
len_train
test_df.shape
df = pd.concat([train_df, test_df], axis = 0)
df_dum = pd.get_dummies(df, drop_first = True)
train_dum_df = df_dum[:len_train]
test_dum_df = df_dum[len_train:]
train_dum_df.shape
test_dum_df.shape
excluded_list = ['SalePrice']
X_feature = [ x for x in train_dum_df.columns if x not in excluded_list ]

X = train_dum_df[X_feature]
Y = train_dum_df[['SalePrice']]

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.3, random_state=123)

X_test = test_dum_df[X_feature]
n_estimators_grid = np.arange(20, 35,2)
depth_grid = np.arange(1, 7)
min_samples_leaf_grid = np.arange(10,21,2)
parameters = {'n_estimators': n_estimators_grid, 'max_depth': depth_grid, 'min_samples_leaf':min_samples_leaf_grid}
gridCV = GridSearchCV(RandomForestRegressor(), param_grid=parameters, cv=5, n_jobs=-1)
gridCV.fit(X_train, Y_train)
best_n_estim = gridCV.best_params_['n_estimators']
best_depth = gridCV.best_params_['max_depth']
best_min_samples_leaf = gridCV.best_params_['min_samples_leaf']
print("Random Forest best n_estimator : " + str(best_n_estim))
print("Random Forest best max_depth : " + str(best_depth))
print("Random Forest best min_samples_leaf : " + str(best_min_samples_leaf))
RF_best = RandomForestRegressor(n_estimators=best_n_estim,max_depth=best_depth,min_samples_leaf=best_min_samples_leaf,random_state=123)
RF_best.fit(X_train, Y_train)
Y_pred = RF_best.predict(X_valid)
#print( "Random Forest best accuracy : " + str(np.round(metrics.accuracy_score(Y_valid,Y_pred),3)))
print('RMSE  : ' + str(np.sqrt(metrics.mean_squared_error(Y_valid, Y_pred))))
plt.scatter(Y_valid, Y_pred, color = 'blue')
plt.xlabel('A Prediced Sale Price')
plt.ylabel('A Real Sale Price')
plt.show()
submission = RF_best.predict(X_test)
submission
train_ori = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_ori = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
Final_data = pd.DataFrame(submission, columns = ['SalePrice'])
Final_data
Final_data = pd.concat([test_ori.Id, Final_data.SalePrice], axis = 1)
Final_data.head()
Final_data.to_csv('submission.csv', index = False)
df = df.drop(['OverallCond', 'Condition2', 'RoofMatl', 'Heating', 'Electrical', 'Functional'], axis = 1)
df['SalePrice'] = np.log(df['SalePrice'])
df_dum = pd.get_dummies(df, drop_first = True)

train_dum_df = df_dum[:len_train]
test_dum_df = df_dum[len_train:]
excluded_list = ['SalePrice']
X_feature = [ x for x in train_dum_df.columns if x not in excluded_list ]

X = train_dum_df[X_feature]
Y = train_dum_df[['SalePrice']]

#poly = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)
#X = poly.fit_transform(X)

#X.shape
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.3, random_state=123)

X_test = test_dum_df[X_feature]
n_estimators_grid = np.arange(55, 70,2)
depth_grid = np.arange(14, 20)
min_samples_leaf_grid = np.arange(1,5)
parameters = {'n_estimators': n_estimators_grid, 'max_depth': depth_grid, 'min_samples_leaf':min_samples_leaf_grid}
gridCV = GridSearchCV(RandomForestRegressor(), param_grid=parameters, cv=5, n_jobs=-1)
gridCV.fit(X_train, Y_train)
best_n_estim = gridCV.best_params_['n_estimators']
best_depth = gridCV.best_params_['max_depth']
best_min_samples_leaf = gridCV.best_params_['min_samples_leaf']
print("Random Forest best n_estimator : " + str(best_n_estim))
print("Random Forest best max_depth : " + str(best_depth))
print("Random Forest best min_samples_leaf : " + str(best_min_samples_leaf))
RF_best = RandomForestRegressor(n_estimators=best_n_estim,max_depth=best_depth,min_samples_leaf=best_min_samples_leaf,random_state=123)
RF_best.fit(X_train, Y_train)
Y_pred = RF_best.predict(X_valid)
#print( "Random Forest best accuracy : " + str(np.round(metrics.accuracy_score(Y_valid,Y_pred),3)))
print('RMSE  : ' + str(np.sqrt(metrics.mean_squared_error(Y_valid, Y_pred))))
submission = RF_best.predict(X_test)
submission
train_ori = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_ori = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
Final_data = pd.DataFrame(np.exp(submission), columns = ['SalePrice'])
Final_data = pd.concat([test_ori.Id, Final_data.SalePrice], axis = 1)
Final_data.to_csv('submission.csv', index = False)
Final_data
plt.scatter(train_df['ScreenPorch'], train_df.LotArea, color = 'blue')
plt.show()
train_df.BsmtCond.value_counts()
sns.boxplot(train_df.BsmtCond, train_df.SalePrice)
sns.boxplot(train_df.BsmtQual, train_df.SalePrice)
df = pd.concat([train_df, test_df], axis = 0)
df = pd.DataFrame(df, columns = train_df.columns)
df
df['BsmtFinSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
df['YearBltAddRemod'] = df['YearBuilt'] + df['YearRemodAdd']
df['Bathroom'] = df['BsmtFullBath'] + df['BsmtHalfBath']*0.5 + df['FullBath'] + df['HalfBath']*0.5
df['GarageAreaPerCar'] = df['GarageArea'] / df['GarageCars']
df['GarageAreaPerCar'].fillna(0, inplace = True)
df = df.drop(['LandSlope', 'ExterCond', 'GarageCond', 'BsmtFinType2', 'KitchenAbvGr', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'YearBuilt', 'YearRemodAdd'], axis = 1)
df = df.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'GarageYrBlt', 'Exterior2nd', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis = 1)
df = df.drop(['GarageArea', 'GarageCars', 'BsmtCond'], axis = 1)
df['SalePrice'] = np.log(df['SalePrice'])
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
feature_num = df.select_dtypes(exclude = ["object"]).columns
feature_num
from scipy.stats import skew 

skew_data = []

for i in feature_num:
    skew_data.append(skew(df[i]))

skew_data = pd.Series(skew_data, index = feature_num)
skew_data.sort_values()
for i in feature_num:
    if (abs(skew_data[i]) > 0.5):
        df[i] = np.log(df[i]+1)
df
df_dum = pd.get_dummies(df, drop_first = True)

train_dum_df = df_dum[:len_train]
test_dum_df = df_dum[len_train:]
excluded_list = ['SalePrice']
X_feature = [ x for x in train_dum_df.columns if x not in excluded_list ]

X = train_dum_df[X_feature]
Y = train_dum_df[['SalePrice']]

# poly = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)
# X = poly.fit_transform(X)

# X.shape
X.shape
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.3, random_state=123)

X_test = test_dum_df[X_feature]
# X_test = poly.fit_transform(X_test)
n_estimators_grid = np.arange(55, 70,2)
depth_grid = np.arange(13, 18)
min_samples_leaf_grid = np.arange(1,5)
parameters = {'n_estimators': n_estimators_grid, 'max_depth': depth_grid, 'min_samples_leaf':min_samples_leaf_grid}
gridCV = GridSearchCV(RandomForestRegressor(), param_grid=parameters, cv=5, n_jobs=-1)
gridCV.fit(X_train, Y_train)
best_n_estim = gridCV.best_params_['n_estimators']
best_depth = gridCV.best_params_['max_depth']
best_min_samples_leaf = gridCV.best_params_['min_samples_leaf']
print("Random Forest best n_estimator : " + str(best_n_estim))
print("Random Forest best max_depth : " + str(best_depth))
print("Random Forest best min_samples_leaf : " + str(best_min_samples_leaf))
RF_best = RandomForestRegressor()
#RF_best = RandomForestRegressor(n_estimators=best_n_estim,max_depth=best_depth,min_samples_leaf=best_min_samples_leaf,random_state=1234)
RF_best.fit(X_train, Y_train)
Y_pred = RF_best.predict(X_valid)
#print( "Random Forest best accuracy : " + str(np.round(metrics.accuracy_score(Y_valid,Y_pred),3)))
print('RMSE  : ' + str(np.sqrt(metrics.mean_squared_error(Y_valid, Y_pred))))
submission = RF_best.predict(X_test)
submission
train_ori = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_ori = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
Final_data = pd.DataFrame(np.exp(submission), columns = ['SalePrice'])
Final_data = pd.concat([test_ori.Id, Final_data.SalePrice], axis = 1)
Final_data.to_csv('submission.csv', index = False)
Final_data
XGB_best = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
param = {
    'max_depth':[2,3,4],
    'n_estimators':range(550,700,50),
    'colsample_bytree':[0.5,0.7,1],
    'colsample_bylevel':[0.5,0.7,1],
}

gridCV = GridSearchCV(XGBRegressor(), param_grid=param, cv=5, n_jobs=-1)
gridCV.fit(X_train, Y_train)
best_n_estim = gridCV.best_params_['n_estimators']
best_depth = gridCV.best_params_['max_depth']
best_colsample_bytree = gridCV.best_params_['colsample_bytree']
best_colsample_bylevel = gridCV.best_params_['colsample_bylevel']
print("XGBoost best n_estimator : " + str(best_n_estim))
print("XGBoost best max_depth : " + str(best_depth))
print("XGBoost best colsample_bytree : " + str(best_colsample_bytree))
print("XGBoost best colsample_bylevel : " + str(best_colsample_bylevel))
#XGB_best = XGBRegressor(n_estimators=best_n_estim,max_depth=best_depth,colsample_bytree=best_colsample_bytree,colsample_bylevel = best_colsample_bylevel,random_state=123)
XGB_best.fit(X_train, Y_train)
Y_pred = XGB_best.predict(X_valid)
#print( "Random Forest best accuracy : " + str(np.round(metrics.accuracy_score(Y_valid,Y_pred),3)))
print('RMSE  : ' + str(np.sqrt(metrics.mean_squared_error(Y_valid, Y_pred))))
np.exp(0.11776782510332362)
submission = XGB_best.predict(X_test)
submission
train_ori = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_ori = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
Final_data = pd.DataFrame(np.exp(submission), columns = ['SalePrice'])
Final_data = pd.concat([test_ori.Id, Final_data.SalePrice], axis = 1)
Final_data.to_csv('submission.csv', index = False)
Final_data
plt.figure(figsize=(20,20))
sns.heatmap(data = train_df.corr(), annot = True, fmt = '.2f', linewidth = .7)