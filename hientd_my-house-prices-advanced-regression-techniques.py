import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time

from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from matplotlib import pyplot as plt
from scipy import stats
from scipy.special import boxcox1p

train_path = '../input/train.csv'
test_path = '../input/test.csv'

def get_RMSE_cv(model, X, y=None, fit_params=None):
    '''
    function to calculate RMSE 
    '''
    return np.sqrt(-cross_val_score(model, X, y,
                      scoring='neg_mean_squared_error',
                      cv=5,
                      fit_params=fit_params)
               )

def helper_RMSE(model):
    return get_RMSE_cv(model, X, y)

# load data 
train_data = pd.read_csv(train_path)
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
test_data = pd.read_csv(test_path)

test_Id = test_data['Id']
test_data.drop('Id', axis=1, inplace=True)

train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)
X = train_data.drop(['Id', 'SalePrice'], axis=1)

y = train_data.SalePrice

print("shape of train: {}\nshape of test: {}".format(train_data.shape, test_data.shape))
print("Short description of Train data:\n")
# plot target to know how values are distributed
sns.distplot(y, fit=stats.norm)
plt.figure()
stats.probplot(y, plot=plt)
# the plot give an insight that the target is right skew
# do log transform
y = np.log1p(y)
# replot to check the distribution
sns.distplot(y, fit=stats.norm)
plt.figure()
stats.probplot(y, plot=plt)
ntrain = X.shape[0]
all_data = pd.concat((X, test_data)).reset_index(drop=True)
# Handling missing values

# fill numerical features
for c in ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
           'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars','GarageArea']:
    all_data[c].fillna(0, inplace=True)

# fill categorical with most frequency value
for c in ['MSZoning', 'Exterior1st', 'Exterior2nd', 'Electrical',
            'KitchenQual', 'Functional', 'SaleType']:
    all_data[c].fillna(all_data[c].mode()[0], inplace=True)
all_data.drop('Utilities', axis=1, inplace=True)
# fill categorical with None value
for c in ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
            'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
    all_data[c].fillna('None', inplace=True)
    
# fill 'LotFrontage' with nearest neighborhood value
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
    
# verifying all_data have no missing data
print(all_data.isnull().any().sum())
# there features are actually categorical
all_data['OverallQual'] = all_data['OverallQual'].apply(str) 
all_data['OverallCond'] = all_data['OverallCond'].apply(str)
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['YrSold'] = all_data['YrSold'].apply(str)
all_data['MoSold'] = all_data['MoSold'].apply(str)

# label encoding
# only apply to categorical have order values
encoder = LabelEncoder()
for c in ['OverallQual', 'OverallCond']:
    all_data[c] = encoder.fit_transform(all_data[c])
    
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
lamda=0.15
for c in all_data.select_dtypes(include=[np.number]).columns:
    all_data[c] = boxcox1p(all_data[c].tolist(), lamda)
# get dummies
all_data = pd.get_dummies(all_data)
all_data.shape
# split to get back train and test set
X = all_data[:ntrain]
test_data = all_data[ntrain:]

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = helper_RMSE(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = helper_RMSE(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = helper_RMSE(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_split=10, 
                                   loss='huber', random_state =5)
score = helper_RMSE(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Kernel Ridge score: 0.1188 (0.0054)
# Lasso score: 0.1153 (0.0066)
# ElasticNet score: 0.1154 (0.0068)
# Gradient Boosting score: 0.1173 (0.0078)
LassoMd = lasso.fit(X, y)
ENetMd = ENet.fit(X, y)
KRRMd = KRR.fit(X, y)
GBoostMd = GBoost.fit(X, y)
finalMd = ( np.expm1(LassoMd.predict(test_data)) + np.expm1(ENetMd.predict(test_data)) 
           + np.expm1(KRRMd.predict(test_data)) + np.expm1(GBoostMd.predict(test_data))
             ) / 4
finalMd
# output to submission file
output = pd.DataFrame({'Id': test_Id,
                      'SalePrice': finalMd})
output.to_csv('submission.csv', index=False)