from sklearn.metrics import mean_absolute_error

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.base import clone

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor
from xgboost import train
from xgboost import DMatrix
from xgboost import cv

from lightgbm import LGBMRegressor

from scipy.stats import norm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

pd.set_option('display.max_rows', 500)
# Read the data
X = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_submit = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)
sns.distplot(y)
y.skew()
y_log = np.log(y)
sns.distplot(y_log)
print("Skewness: %f" % y_log.skew())
features = pd.concat([X, X_submit]).reset_index(drop=True)
features.shape
all_data_na = (features.isnull().sum() / len(features)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
features["PoolQC"] = features["PoolQC"].fillna("None")
features["MiscFeature"] = features["MiscFeature"].fillna("None")
features["Alley"] = features["Alley"].fillna("None")
features["Fence"] = features["Fence"].fillna("None")
features["FireplaceQu"] = features["FireplaceQu"].fillna("None")
nans_cols = []

for k in features:
    if features[k].isnull().sum() > 0:
        nans_cols.append((k, features[k].isnull().sum()))
nans_cols = sorted(nans_cols, key = lambda t: t[1], reverse=True)        
nans_cols
garage_X = features[features['GarageYrBlt'].isnull()][['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageArea']]
garage_X[garage_X.notnull()]
features['has_garage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
garage_X = features[features['BsmtCond'].isnull()][['BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'TotalBsmtSF']]
garage_X[garage_X.notnull()]
features['MasVnrArea'] = features['MasVnrArea'].fillna(0)
features['MasVnrType'] = features['MasVnrType'].fillna('None')
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].value_counts().idxmax())
features['Utilities'] = features['Utilities'].fillna(features['Utilities'].value_counts().idxmax())
features['BsmtFullBath'] = features['BsmtFullBath'].fillna(features['BsmtFullBath'].value_counts().idxmax())
features['Functional'] = features['Functional'].fillna(features['Functional'].value_counts().idxmax())
features['Exterior1st'] = features['Exterior1st'].fillna(features['Functional'].value_counts().idxmax())
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].value_counts().idxmax())
features['BsmtFinSF1'] = features['BsmtFinSF1'].fillna(features['BsmtFinSF1'].value_counts().idxmax())
features['BsmtFinSF2'] = features['BsmtFinSF2'].fillna(0)
features['BsmtUnfSF'] = features['BsmtUnfSF'].fillna(0)
features['BsmtHalfBath'] = features['BsmtHalfBath'].fillna(features['BsmtHalfBath'].value_counts().idxmax())
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].value_counts().idxmax())
features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].value_counts().idxmax())
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].value_counts().idxmax())

# GarageType etc : data description says NA for garage features is "no garage"
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')

# BsmtQual etc : data description says NA for basement features is "no basement"
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
features["LotFrontage"] = features.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
#Check remaining missing values if any 
features_na = (features.isnull().sum() / len(features)) * 100
features_na = features_na.drop(features_na[features_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :features_na})
missing_data.head()
# Some numerical features are actually really categories
features = features.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })
features.shape
final_features = pd.get_dummies(features).reset_index(drop=True)
final_features.shape
X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]
X.shape, y.shape, X_sub.shape
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.3, random_state=42)
#Validation function
n_folds = 5

def rmsle_cv(model, X, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse= np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
xgb.fit(X_train, y_train)
print("Accuracy score: {0}".format(xgb.score(X_test, y_test)))

predictions_regr = np.exp(xgb.predict(X_test))
print("Mean Absolute Error: {0}".format(mean_absolute_error(np.exp(y_test), predictions_regr)))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso.fit(X_train, y_train)
print("Accuracy score: {0}".format(lasso.score(X_test, y_test)))

predictions_regr = np.exp(lasso.predict(X_test))
print("Mean Absolute Error: {0}".format(mean_absolute_error(np.exp(y_test), predictions_regr)))
lgb = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lgb.fit(X_train, y_train)
print("Accuracy score: {0}".format(lgb.score(X_test, y_test)))

predictions_regr = np.exp(lgb.predict(X_test))
print("Mean Absolute Error: {0}".format(mean_absolute_error(np.exp(y_test), predictions_regr)))
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
gboost.fit(X_train, y_train)
print("Accuracy score: {0}".format(gboost.score(X_test, y_test)))

predictions_regr = np.exp(gboost.predict(X_test))
print("Mean Absolute Error: {0}".format(mean_absolute_error(np.exp(y_test), predictions_regr)))
score = rmsle_cv(lasso, X_train, y_train)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(xgb, X_train, y_train)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(lgb, X_train, y_train)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
score = rmsle_cv(gboost, X_train, y_train)
print("Gradient Boosting score score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
gboost.fit(X, y_log)
submission = np.exp(gboost.predict(X_sub))
sub = pd.DataFrame({'Id': X_submit.index,'SalePrice': submission})

sub.to_csv('iowa_houses_submission_gboost.csv', sep=',', index=False)