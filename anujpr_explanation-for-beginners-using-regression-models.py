# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)

train.shape, test.shape
train.describe().T #transposes the actual describe
sns.distplot(train['SalePrice']);
#correlation matrix



corrmat = train.corr()

f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

#k = 10 #number of variables for heatmap



cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#Graph for SalePrice v/s OverallQual



var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Graph for SalePrice v/s GrLivArea



var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Graph for SalePrice v/s GrLivArea after deleting outliers



var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Graph for SalePrice v/s GarageCars



var = 'GarageCars'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Graph for SalePrice v/s GarageArea



var = 'GarageArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Graph for SalePrice v/s TotalBsmtSF



var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Graph for SalePrice v/s 1stFlrSF



var = '1stFlrSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Graph for SalePrice v/s FullBath



var = 'FullBath'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#Graph for SalePrice v/s TotRmsAbvGrd



var = 'TotRmsAbvGrd'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
train["SalePrice"] = np.log1p(train["SalePrice"])

y = train['SalePrice'].reset_index(drop=True)



sns.distplot(train['SalePrice']);
train.shape, test.shape
combine = pd.concat([train, test], sort = False).reset_index(drop=True)

combine.drop(['SalePrice'], axis=1, inplace=True)

print("Size of combined data set is : {}".format(combine.shape))
combine.describe()
def miss_perc(df):

  df_null_data = (df.isnull().sum() / len(combine)) * 100

  df_null_data = df_null_data.drop(df_null_data[df_null_data == 0].index).sort_values(ascending=False)[:30]

  return pd.DataFrame({'Missing Percentage' :df_null_data})



miss_perc(combine)
combine['MSSubClass'] = combine['MSSubClass'].apply(str)

combine['YrSold'] = combine['YrSold'].astype(str)

combine['MoSold'] = combine['MoSold'].astype(str)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 

            'BsmtHalfBath','GarageYrBlt', 'GarageArea','GarageCars','MasVnrArea'):

    combine[col] = combine[col].fillna(0)



for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

            'Fence','PoolQC','MiscFeature','Alley','FireplaceQu','Fence','GarageType',

            'GarageFinish', 'GarageQual', 'GarageCond']:

    combine[col] = combine[col].fillna('None')



for col in ['Utilities','Exterior1st','Exterior2nd','SaleType','Functional','Electrical',

            'KitchenQual', 'GarageFinish', 'GarageQual', 'GarageCond','MasVnrType']:

    combine[col] = combine[col].fillna(combine[col].mode()[0])



combine['LotFrontage'] = combine.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



combine['MSZoning'] = combine.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



miss_perc(combine)
categorical_features = combine.dtypes[combine.dtypes == "object"].index



combine.update(combine[categorical_features].fillna('None'))



categorical_features
numerical_features = combine.dtypes[combine.dtypes != "object"].index



combine.update(combine[numerical_features].fillna(0))



numerical_features
from scipy import stats

from scipy.stats import norm, skew, boxcox_normmax # for statistics

from scipy.special import boxcox1p
skewed_features = combine[numerical_features].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_features
high_skew_feat = skewed_features[abs(skewed_features) > 0.5]

skewed_features = high_skew_feat.index



for feature in skewed_features:

  combine[feature] = boxcox1p(combine[feature], boxcox_normmax(combine[feature] + 1))
combine = combine.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



combine['TotalSF'] = combine['TotalBsmtSF'] + combine['1stFlrSF'] + combine['2ndFlrSF']



combine['YrBltAndRemod'] = combine['YearBuilt']+ combine['YearRemodAdd']



combine['Total_sqr_footage'] = (combine['BsmtFinSF1'] + combine['BsmtFinSF2'] + combine['1stFlrSF'] + combine['2ndFlrSF'])



combine['Total_Bathrooms'] = (combine['FullBath'] + (0.5 * combine['HalfBath']) + combine['BsmtFullBath'] + (0.5 * combine['BsmtHalfBath']))



combine['Total_porch_sf'] = (combine['OpenPorchSF'] + combine['3SsnPorch'] + combine['EnclosedPorch'] + combine['ScreenPorch'] +

                             combine['WoodDeckSF'])



combine['haspool'] = combine['PoolArea'].apply(lambda x: 1 if x > 0 else 0)



combine['has2ndfloor'] = combine['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)



combine['hasgarage'] = combine['GarageArea'].apply(lambda x: 1 if x > 0 else 0)



combine['hasbsmt'] = combine['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)



combine['hasfireplace'] = combine['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(combine[c].values)) 

    combine[c] = lbl.transform(list(combine[c].values))
combine = pd.get_dummies(combine)

print(combine.shape)
X = combine.iloc[:len(y), :]

X_sub = combine.iloc[len(y):, :]
X.shape, y.shape, X_sub.shape
overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.94:

        overfit.append(i)



overfit = list(overfit)



X = X.drop(overfit, axis=1).copy()

X_sub = X_sub.drop(overfit, axis=1).copy()

overfit
from datetime import datetime



from sklearn.linear_model import ElasticNetCV, Lasso, ElasticNet, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

from sklearn.svm import SVR



from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error



from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor



import sklearn.linear_model as linear_model
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
ENet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
XGBoostR = XGBRegressor(learning_rate=0.01,n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.7, 

                       objective='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=0.00006, silent = True)
CatBoostR = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=10, eval_metric='RMSE', random_seed = 42,

                        bagging_temperature = 0.2, od_type='Iter', metric_period = 50, od_wait=20)
LightGBMR = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75,

                                       bagging_freq=5, bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7, verbose=-1, )
StackCVR_gen = StackingCVRegressor(regressors=(ridge, lasso, ENet, CatBoostR, XGBoostR, LightGBMR), 

                                meta_regressor=XGBoostR, use_features_in_secondary=True)
# Using various prediction models that we just created 



score = cv_rmse(ridge , X)

print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lasso , X)

print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(ENet)

print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(svr)

print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(LightGBMR)

print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(CatBoostR)

print("catboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(XGBoostR)

print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
print('START Fit')



print('stack_gen')

stack_gen_model = StackCVR_gen.fit(np.array(X), np.array(y))



print('elasticnet')

elastic_model_full_data = ENet.fit(X, y)



print('Lasso')

lasso_model_full_data = lasso.fit(X, y)



print('Ridge')

ridge_model_full_data = ridge.fit(X, y)



print('Svr')

svr_model_full_data = svr.fit(X, y)



print('catboost')

cbr_model_full_data = CatBoostR.fit(X, y)



print('xgboost')

xgb_model_full_data = XGBoostR.fit(X, y)



print('lightgbm')

lgb_model_full_data = LightGBMR.fit(X, y)
def blend_models_predict(X):

    return ((0.15 * elastic_model_full_data.predict(X)) + \

            (0.15 * lasso_model_full_data.predict(X)) + \

            (0.1 * ridge_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

            (0.05 * cbr_model_full_data.predict(X)) + \

            (0.05 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))
print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))
print('Predict submission')

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
print('Blend with Top Kernels submissions\n')

sub_1 = pd.read_csv('https://raw.githubusercontent.com/AnujPR/Kaggle-Hybrid-House-Prices-Prediction/master/masum_rumia-detailed-regression-guide-with-house-pricing%20submission.csv')

sub_2 = pd.read_csv('https://raw.githubusercontent.com/AnujPR/Kaggle-Hybrid-House-Prices-Prediction/master/serigne_stacked-regressions-top-4-on-leaderboard_submission.csv')

sub_3 = pd.read_csv('https://raw.githubusercontent.com/AnujPR/Kaggle-Hybrid-House-Prices-Prediction/master/jesucristo1-house-prices-solution-top-1_new_submission.csv')

submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 

                                (0.25 * sub_1.iloc[:,1]) + 

                                (0.25 * sub_2.iloc[:,1]) + 

                                (0.25 * sub_3.iloc[:,1]))
q1 = submission['SalePrice'].quantile(0.0042)

q2 = submission['SalePrice'].quantile(0.99)

# Quantiles helping us get some extreme values for extremely low or high values 

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission.csv", index=False)
submission.head()