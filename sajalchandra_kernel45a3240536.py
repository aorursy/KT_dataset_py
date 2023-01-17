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
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os
print(os.listdir("../input"))
from datetime import datetime
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.sample(10)
train.shape, test.shape
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
train['SalePrice'].hist(bins=50)
train.GrLivArea.sort_values()
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)
train['SalePrice'] = np.log1p(train['SalePrice'])
y = train['SalePrice'].reset_index(drop=True)
train['SalePrice'].hist(bins=50)
y
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)
features.shape
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

features['Functional'] = features['Functional'].fillna('Typ')
features['Electrical'] = features['Electrical'].fillna("SBrkr") 
features['KitchenQual'] = features['KitchenQual'].fillna("TA") 
features["PoolQC"] = features["PoolQC"].fillna("None")



## Filling these with MODE , i.e. , the most frequent value in these columns .
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    features[col] = features[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
obj = []
for i in features.columns:
    if features[i].dtype == object:
        obj.append(i)
features.update(features[obj].fillna('None'))
print(obj)
features.groupby('Neighborhood')['LotFrontage'].mean()
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics=[]
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics.append(i)
features.update(features[numerics].fillna(0))

len(numerics)
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i]+1))
features = features.drop(['Utilities', 'Street', 'PoolQC'], axis=1)

features['YrBltAndRemod'] = features['YearBuilt']+features['YearRemodAdd']
features['TotalSF'] = features['TotalBsmtSF']+features['1stFlrSF']+features['2ndFlrSF']
features['Total_sqr_footage'] = features['BsmtFinSF1']+features['BsmtFinSF2']+features['1stFlrSF']+features['2ndFlrSF']
features['Total_Bathrooms'] = (features['FullBath']+0.5*features['HalfBath'])+features['BsmtFullBath']+(0.5*features['BsmtHalfBath'])
features['Total_porch_SF'] = features['OpenPorchSF']+features['3SsnPorch']+features['EnclosedPorch']+features['ScreenPorch']+features['WoodDeckSF']
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x>0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x>0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x>0 else 0)

features.shape
features.head()
final_features = pd.get_dummies(features).reset_index(drop=True)
final_features.shape
final_features.sample(10)
x = final_features.iloc[:len(y), :]
x_sub = final_features.iloc[len(y):, :]
x.shape, y.shape, x_sub.shape
outliers = [30, 88, 462, 631, 1322]
x = x.drop(x.index[outliers])
y = y.drop(y.index[outliers])

overfit = []
for i in x.columns:
    counts = x[i].value_counts()
    zeros = counts.iloc[0]
    if zeros/len(x)*100 > 99.94:
        overfit.append(i)
print('Overfitting Columns: ', overfit)

overfit = list(overfit)
x = x.drop(overfit, axis=1)
x_sub = x_sub.drop(overfit, axis=1)
overfit

x.shape, x_sub.shape, y.shape
kfolds = KFold(n_splits=10, shuffle=True)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, x=x):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=kfolds))
    return rmse

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber')
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm), meta_regressor=xgboost, use_features_in_secondary=True)

score = cv_rmse(ridge , x)
score = cv_rmse(lasso , x)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(elasticnet)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
print('START Fit')

print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(x), np.array(y))

print('elasticnet')
elastic_model_full_data = elasticnet.fit(x, y)

print('Lasso')
lasso_model_full_data = lasso.fit(x, y)

print('Ridge')
ridge_model_full_data = ridge.fit(x, y)

print('Svr')
svr_model_full_data = svr.fit(x, y)

print('GradientBoosting')
gbr_model_full_data = gbr.fit(x, y)

print('xgboost')
xgb_model_full_data = xgboost.fit(x, y)

print('lightgbm')
lgb_model_full_data = lightgbm.fit(x, y)
def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) + \
            (0.05 * lasso_model_full_data.predict(X)) + \
            (0.1 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))
    
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(x)))
print('Predict Submission')
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission.iloc[:,1] = (np.expm1(blend_models_predict(x_sub)))
