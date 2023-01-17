import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.pipeline import make_pipeline
#
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

prefix_path = '../input/'
train_df = pd.read_csv(prefix_path+'train.csv') # 0 ~ 1459
test_df = pd.read_csv(prefix_path+'test.csv') # 1460 ~ 2919
dataset_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
# make 'SalePrice' be the last columns
temp = dataset_df['SalePrice']
temp1 = dataset_df.drop(['SalePrice'], axis=1).sort_index(axis=1)
dataset_df = pd.concat([temp1,temp], axis=1)
del temp, temp1
dataset_df.head()
# heatmap with correlation matrix
# multicollinearity可能造成模型預測失真，透過heatmap可以進行初步的篩選
corr = dataset_df.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, square=True, cmap='rainbow')
# check k most correlated features with 'SalePrice'
# 彼此相關性太高的就不要同時選
# 時間序列的特徵要多考慮一下
k_largest_features = corr.nlargest(10, 'SalePrice').index
sns.heatmap(dataset_df[k_largest_features].corr(), square=True, cmap='rainbow', annot=True)
selected_numerical_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice']
sns.pairplot(train_df[selected_numerical_features])
fig, [ax, ax1, ax2] = plt.subplots(1,3)
fig.set_size_inches(16, 4)
sns.distplot(dataset_df['SalePrice'].dropna(), label='skewness:%.2f'%dataset_df['SalePrice'].skew(), ax=ax).legend(loc='best')
sns.distplot(dataset_df['GrLivArea'].dropna(), label='skewness:%.2f'%dataset_df['GrLivArea'].skew(), ax=ax1).legend(loc='best')
sns.distplot(dataset_df['TotalBsmtSF'].dropna(), label='skewness:%.2f'%dataset_df['TotalBsmtSF'].skew(), ax=ax2).legend(loc='best')
# RoofStyle
print('if exist missing values:', dataset_df['RoofStyle'].isnull().any())
sns.factorplot(x='RoofStyle', y='SalePrice', data=dataset_df, kind='box')
# MSZoning
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='MSZoning', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='MSZoning',data=dataset_df, ax=ax1)
plt.close(g.fig)
# BldgType...好像沒什麼影響
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='BldgType', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='BldgType',data=dataset_df, ax=ax1)
plt.close(g.fig)
# HouseStyle
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='HouseStyle', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='HouseStyle',data=dataset_df, ax=ax1)
plt.close(g.fig)
# CentralAir
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='CentralAir', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='CentralAir',data=dataset_df, ax=ax1)
plt.close(g.fig)
# ExterQual
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='ExterQual', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='ExterQual',data=dataset_df, ax=ax1)
plt.close(g.fig)
# SaleType...沒啥影響(因為WD太多了)
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='SaleType', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='SaleType',data=dataset_df, ax=ax1)
plt.close(g.fig)
# LandSlope...沒啥影響(因為Gtl太多了)
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='LandSlope', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='LandSlope',data=dataset_df, ax=ax1)
plt.close(g.fig)
# ExterCond...沒啥影響(TA太多)
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='ExterCond', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='ExterCond',data=dataset_df, ax=ax1)
plt.close(g.fig)
# Foundation
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='Foundation', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='Foundation',data=dataset_df, ax=ax1)
plt.close(g.fig)
# PavedDrive
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='PavedDrive', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='PavedDrive',data=dataset_df, ax=ax1)
plt.close(g.fig)
# GarageFinish
fig, [ax, ax1] = plt.subplots(1, 2)
fig.set_size_inches(12, 4)
g = sns.factorplot(x='GarageFinish', y='SalePrice', data=dataset_df, kind='box', ax=ax)
sns.countplot(x='GarageFinish',data=dataset_df, ax=ax1)
plt.close(g.fig)
# GrLivArea
dataset_df.plot.scatter('GrLivArea', 'SalePrice')
total_outlier_num = 0
outlier_num = len(dataset_df[(dataset_df['GrLivArea']>4000) & (dataset_df['SalePrice']<300000)])
total_outlier_num += outlier_num
dataset_df.drop(dataset_df[(dataset_df['GrLivArea']>4000) & (dataset_df['SalePrice']<300000)].index, inplace=True)
dataset_df.plot.scatter('GrLivArea', 'SalePrice')
print('delete %d data' % outlier_num)
#numeric_features = dataset_df.select_dtypes(exclude='object').columns
#feature_skewness = dataset_df[numeric_features].skew().sort_values()
#for feature, skewness in zip(feature_skewness.index, feature_skewness):
#    #print(feature, abs(skewness))
#    if(abs(skewness) > 0.8):
#        dataset_df[feature] = dataset_df[feature].apply(lambda i: np.log1p(i) if i >= 0 else np.nan)
dataset_df['SalePrice'] = dataset_df['SalePrice'].apply(lambda i: np.log1p(i) if i >= 0 else np.nan)
sns.distplot(dataset_df['SalePrice'].dropna(), label='skewness:%.2f'%dataset_df['SalePrice'].skew()).legend(loc='best')
sorted_missing_feature_df = pd.DataFrame({'count': dataset_df.isnull().sum().sort_values(ascending=False)})

print(sorted_missing_feature_df[sorted_missing_feature_df['count']>0])
print('total:', len(sorted_missing_feature_df[sorted_missing_feature_df['count']>0]), 'missing values')
# total 35 features have missing value
exist_missing_feature_df = sorted_missing_feature_df[sorted_missing_feature_df['count']>0]
fig, ax = plt.subplots()
plt.xticks(rotation='75')
fig.set_size_inches(15,9)
sns.barplot(x=exist_missing_feature_df.index,y=exist_missing_feature_df['count'])
# PoolQC
# NaN means no pool
dataset_df['PoolQC'].fillna('None', inplace=True)
# MiscFeature
# NaN means no MiscFeature
dataset_df['MiscFeature'].fillna('None', inplace=True)
# Alley
# NaN means no Alley access
dataset_df['Alley'].fillna('None', inplace=True)
# Fence
# NaN means no Fence
dataset_df['Fence'].fillna('None', inplace=True)
# FireplaceQu
# NaN means no Fireplace
dataset_df['FireplaceQu'].fillna('None', inplace=True)
# LotFrontage
# fill missing value depends on LotFrontage groupy by neighborhood median
dataset_df['LotFrontage'] = dataset_df.groupby('Neighborhood')['LotFrontage'].transform(lambda i: i.fillna(i.median()))
# GarageFinish GarageQual GarageType GarageCond
# NaN means no Garage
dataset_df['GarageFinish'].fillna('None', inplace=True)
dataset_df['GarageQual'].fillna('None', inplace=True)
dataset_df['GarageType'].fillna('None', inplace=True)
dataset_df['GarageCond'].fillna('None', inplace=True)

# GarageYrBlt GarageArea GarageCars
# NaN means no Garage
dataset_df['GarageYrBlt'].fillna(0, inplace=True)
dataset_df['GarageArea'].fillna(0, inplace=True)
dataset_df['GarageCars'].fillna(0, inplace=True)
# BsmtFinSF1 BsmtFinSF2 BsmtUnfSF TotalBsmtSF BsmtFullBath BsmtHalfBath 
# NaN means no basement
dataset_df['BsmtFinSF1'].fillna(0, inplace=True)
dataset_df['BsmtFinSF2'].fillna(0, inplace=True)
dataset_df['BsmtUnfSF'].fillna(0, inplace=True)
dataset_df['TotalBsmtSF'].fillna(0, inplace=True)
dataset_df['BsmtFullBath'].fillna(0, inplace=True)
dataset_df['BsmtHalfBath'].fillna(0, inplace=True)

# BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2
# NaN means no basement
dataset_df['BsmtQual'].fillna('None', inplace=True)
dataset_df['BsmtCond'].fillna('None', inplace=True)
dataset_df['BsmtExposure'].fillna('None', inplace=True)
dataset_df['BsmtFinType1'].fillna('None', inplace=True)
dataset_df['BsmtFinType2'].fillna('None', inplace=True)
# MasVnrArea MasVnrType 
# NaN means no masonry veneer
dataset_df['MasVnrArea'].fillna(0, inplace=True)
dataset_df['MasVnrType'].fillna('None', inplace=True)
# Functional  
# fill with most common value
dataset_df['Functional'].fillna(dataset_df['Functional'].mode()[0], inplace=True)
# Utilities
# data description says NaN means typical
dataset_df['Utilities'].fillna('Typ', inplace=True)
# Electrical 
# fill with most common value
dataset_df['Electrical'].fillna(dataset_df['Electrical'].mode()[0], inplace=True)
# KitchenQual 
# fill with most common value
dataset_df['KitchenQual'].fillna(dataset_df['KitchenQual'].mode()[0], inplace=True)
# Exterior1st Exterior2nd
# fill with most common value
dataset_df['Exterior1st'].fillna(dataset_df['Exterior1st'].mode()[0], inplace=True)
dataset_df['Exterior2nd'].fillna(dataset_df['Exterior2nd'].mode()[0], inplace=True)
# SaleType 
# fill with most common value
dataset_df['SaleType'].fillna(dataset_df['SaleType'].mode()[0], inplace=True)
# MSSubClass 
# NaN means no type
dataset_df['MSSubClass'].fillna('None', inplace=True)
# MSZoning
# fill with most common value
dataset_df['MSZoning'].fillna(dataset_df['MSZoning'].mode()[0], inplace=True)
dataset_df.isnull().sum().sort_values(ascending=False).head()

# Transforming some numerical variables that are really categorical
dataset_df['MSSubClass'] = dataset_df['MSSubClass'].apply(str)
dataset_df['YrSold'] = dataset_df['YrSold'].apply(str)
dataset_df['MoSold'] = dataset_df['MoSold'].apply(str)
ordinal_features_1 = ['BsmtQual', 'BsmtCond', 'GarageQual', 'FireplaceQu', 'GarageCond',
                 'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual']
ordinal_features_2 = ['BsmtFinType1', 'BsmtFinType2']
for feature in ordinal_features_1:
    dataset_df[feature] = dataset_df[feature].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0})
for feature in ordinal_features_2:
    dataset_df[feature] = dataset_df[feature].map({'GLQ':6, 'ALQ':5, 'BLQ':4,
                                                   'Rec':3, 'LwQ':2, 'Unf':1, 'None':0})
# Fence
# dataset_df['Fence'].unique()
dataset_df['Fence'] = dataset_df['Fence'].map({'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1, 'None':0})
# BsmtExposure
# dataset_df['BsmtExposure'].unique()
dataset_df['BsmtExposure'] = dataset_df['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'None':0})
# GarageFinish
dataset_df['GarageFinish'] = dataset_df['GarageFinish'].map({'RFn':3, 'Unf':2, 'Fin':1, 'None':0})
# LandSlope
#dataset_df['LandSlope'].unique()
dataset_df['LandSlope'] = dataset_df['LandSlope'].map({'Sev':2, 'Mod':1, 'Gtl':0})
# LotShape
dataset_df['LotShape'] = dataset_df['LotShape'].map({'Reg':3, 'IR1':2, 'IR2':1, 'IR3':0})
dataset_df = pd.get_dummies(dataset_df)

# drop 'Id' feature
dataset_df = dataset_df.drop('Id', axis=1)
#dataset_df = dataset_df[selected_features]
train_data = dataset_df[:len(train_df)-total_outlier_num]
test_data = dataset_df[len(train_df)-total_outlier_num:]
X_train = train_data.drop(['SalePrice'], axis=1).values
y_train = train_data['SalePrice'].values
X_test = test_data.drop(['SalePrice'], axis=1).values
model_set = []
#model_set.append(('LinearRegression',LinearRegression()))
model_set.append(('Ridge',Ridge()))
model_set.append(('Lasso',Lasso(alpha=0.0005)))
model_set.append(('LinearSVR',LinearSVR()))
model_set.append(('XGBRegressor',XGBRegressor()))
model_set.append(('LGBMRegressor',LGBMRegressor()))
model_set.append(('RandomForestRegressor', RandomForestRegressor()))

cv_results = []
for _,reg in model_set:
    mse_score = cross_val_score(reg, X_train, y_train.ravel(), scoring='neg_mean_squared_error', cv=10, n_jobs=4)
    rmse_score = np.sqrt(-mse_score) #cross_val_score返回的mse值是負的，變個號就是正確的值(原因詳見github)
    cv_results.append(rmse_score)
cv_means = []
cv_std = []
for cv in cv_results:
    cv_means.append(cv.mean())
    cv_std.append(cv.std())
cv_df = pd.DataFrame({'CrossValMeans':cv_means, 'CrossValStd': cv_std, 'Algo': [name for name,_ in model_set]})
cv_df.sort_values(by='CrossValMeans', inplace=True)
print(cv_df)
sns.barplot('CrossValMeans', 'Algo', data=cv_df, palette='hls', **{'xerr':cv_std})

robust_scaler = RobustScaler()
robust_X_train = robust_scaler.fit_transform(X_train)
robust_X_test = robust_scaler.transform(X_test)
LassoRegression = Lasso(alpha=0.0005).fit(robust_X_train, y_train.ravel())

feature_importance = pd.DataFrame({'FeatureImportance':LassoRegression.coef_, 'Index':train_data.columns[:-1]})
mse_score = cross_val_score(LassoRegression, robust_X_train, y_train.ravel(), scoring='neg_mean_squared_error', cv=10, n_jobs=4)
print('rmse:', np.sqrt(-mse_score).mean()) #cross_val_score返回的mse值是負的，變個號就是正確的值(原因詳見github)


feature_importance.sort_values(by='FeatureImportance', ascending=False, inplace=True)
# select feature_importance != 0 features
feature_importance = feature_importance[feature_importance['FeatureImportance'] != 0]
print(feature_importance)
fig, ax = plt.subplots()
fig.set_size_inches(12,18)
sns.barplot(x='FeatureImportance', y='Index', data=feature_importance, orient='h')
LinearRegression_prediction = LassoRegression.predict(robust_X_test)
LinearRegression_prediction = np.expm1(LinearRegression_prediction)
print(LinearRegression_prediction)
XGB = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1).fit(X_train, y_train)
#XGB
mse_score = cross_val_score(XGB, X_train, y_train.ravel(), scoring='neg_mean_squared_error', cv=5, n_jobs=4)
print('rmse:', np.sqrt(-mse_score).mean()) #cross_val_score返回的mse值是負的，變個號就是正確的值(原因詳見github)
XGB_prediction = XGB.predict(X_test)
XGB_prediction = np.expm1(XGB_prediction)
print(XGB_prediction)
LGB = LGBMRegressor(objective='regression',num_leaves=5,
                    learning_rate=0.05, n_estimators=720,
                    max_bin = 55, bagging_fraction = 0.8,
                    bagging_freq = 5, feature_fraction = 0.2319,
                    feature_fraction_seed=9, bagging_seed=9,
                    min_data_in_leaf =6, min_sum_hessian_in_leaf = 11).fit(X_train, y_train)
#LGB
mse_score = cross_val_score(LGB, X_train, y_train.ravel(), scoring='neg_mean_squared_error', cv=5, n_jobs=4)
print('rmse:', np.sqrt(-mse_score).mean()) #cross_val_score返回的mse值是負的，變個號就是正確的值(原因詳見github)
LGB_prediction = LGB.predict(X_test)
LGB_prediction = np.expm1(LGB_prediction)
print(LGB_prediction)
prediction = 0.3 * XGB_prediction + 0.4 * LinearRegression_prediction + 0.3 * LGB_prediction
submission = pd.DataFrame({'Id':test_df['Id'],'SalePrice':prediction})
submission.to_csv('submission.csv',index=False)
