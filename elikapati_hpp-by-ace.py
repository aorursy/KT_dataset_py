import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime

from scipy import stats

from scipy.stats import skew

from scipy.stats import boxcox_normmax

from scipy.special import boxcox1p

from scipy.special import inv_boxcox



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error



from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR



from sklearn import linear_model

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape)

train.head()
print(test.shape)

test.head()
all_data = pd.concat([train, test], axis=0, sort=False)

print(all_data.shape)

all_data.head()
# Let us verify how the target i.e. SalePrice is distributed



target = train['SalePrice']

sns.distplot(target)
# Measure skewness



print('The skewness of SalePrice is {0:.2f}'.format(stats.skew(target)))
# Pick a transformation to remove skewness (among several popular transformations)



sp_log = np.log1p(target)

print('Skew after Log transformation = {0:.2f}'.format(stats.skew(sp_log)))



target_trnsfmd, lambda_ = stats.boxcox(target)

print('Lambda = ', lambda_)

print('Skew after BoxCox transformation = {0:.2f}'.format(stats.skew(target_trnsfmd)))
# Transform target



train['SalePrice'] = target_trnsfmd
# Let us verify SalePrice distribution after log transformation



sns.distplot(target_trnsfmd)
# Show count and percent of nulls by feature in a dataframe



def getNullStats(df):

    tbl_results = []

    print('Total Features of dataset = ', len(df.columns))



    total_samples = len(df)

    null_samples = df.isnull().sum()



    tbl_results = pd.concat([null_samples, round(null_samples/total_samples*100, 2)], axis=1)

    tbl_results = tbl_results.rename(columns = {0:'Nulls', 1:'Percent'})

    tbl_results = tbl_results[tbl_results.iloc[:, 1] !=0].sort_values('Nulls', ascending=False).round(2)



    print('Null Features of dataset = ', len(tbl_results))

    return tbl_results
# Table of features having missing values from "train" dataset



mvtbl_train = getNullStats(train)

mvtbl_train
# Features of train dataset having missing values



mvfs_train = mvtbl_train.index

mvfs_train
# Table of features having missing values from "test" dataset



mvfstbl_test = getNullStats(test)

mvfstbl_test
# Features of test dataset having missing values



mvfs_test = mvfstbl_test.index

mvfs_test
# Features having missing values from both "train and test" datasets



mvfs_all = mvfs_train | mvfs_test

mvfs_all
drop_feats = ['PoolQC', 'MiscFeature', 'Alley']

train.drop(drop_feats, axis=1, inplace=True)

test.drop(drop_feats, axis=1, inplace=True)

all_data.drop(drop_feats, axis=1, inplace=True)
# Features having missing values from both "train and test" datasets (updated)



mvtbl_train = getNullStats(train)

print()

mvtbl_test  = getNullStats(test)



mvfs_train = mvtbl_train.index

mvfs_test = mvtbl_test.index



mvfs_all = mvfs_train | mvfs_test

mvfs_all
num_feats = train.select_dtypes(include=np.number).columns

print('Numeric Features = ', len(num_feats))

num_feats
cat_feats = train.select_dtypes(exclude=np.number).columns

print('Categoric Features = ', len(cat_feats))

cat_feats
numToCatFeats = ['MSSubClass', 'OverallQual', 'OverallCond']

for feat in numToCatFeats:

    train[feat] = train[feat].astype('object')
num_feats = train.select_dtypes(include=np.number).columns

num_feats = num_feats.drop(['Id', 'SalePrice'])



print('Numeric Features = ', len(num_feats))

num_feats
cat_feats = train.select_dtypes(exclude=np.number).columns

print('Categoric Features = ', len(cat_feats))

cat_feats
# Get categoric features having nulls



cat_null_feats = all_data[cat_feats].isnull().sum() > 0

cat_null_feats = cat_null_feats[(cat_null_feats > 0)].index

cat_null_feats
# Impute 'MSZoning' based on 'MSSubClass'



all_data['MSZoning'] = train.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# Basement Feature set - 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'

# Garage Feature set - 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'

# From Qualitative features, leave Basement feature set and Garage feature set for later imputation

# Impute remaining features with mode



cat_null_sub_feats = ['Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 

                      'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 

                      'Fence', 'SaleType']
# Impute other categoric features having missing values with the "mode" (most frequent) value



for f in cat_null_sub_feats:

    all_data[f] = train[f].fillna(train[f].mode()[0])
# Get numeric features having nulls



num_null_feats = all_data[num_feats].isnull().sum() > 0

num_null_feats = num_null_feats[(num_null_feats > 0)].index

num_null_feats
def anova(frame, indvar):

    anv = pd.DataFrame()

    anv['features'] = cat_feats

    pvals = []

    for c in cat_feats:

           samples = []

           for cls in frame[c].unique():

                  s = frame[frame[c] == cls][indvar].values

                  samples.append(s)

           pval = stats.f_oneway(*samples)[1]

           pvals.append(pval)

    anv['pval'] = pvals

    return anv.sort_values('pval')
# Find 'the feature' which has strong relevance with 'SalePrice'

# Note: The target, 'SalePrice' is most influenced by 'OverallQual' among the categorical features.

# It's value is so small that it is zero with many number of decimals; so I took the 'pval' of Neighborhood and used it

# as our objective is to show relative strength of features on 'SalePrice'.



cat_data = train[cat_feats].copy()

cat_data['SalePrice'] = train['SalePrice']



k = anova(cat_data, 'SalePrice')

min_val = np.sort(k['pval'].values)[1]

k['disparity'] = np.log(1./(k['pval'].values + min_val))

plt.figure(figsize=(16, 4))

sns.barplot(data=k, x='features', y='disparity')

plt.xticks(rotation=90)
# Impute 'LotFrontage' based on 'OverallQual'



all_data['LotFrontage'] = train.groupby('OverallQual')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# Impute remaining numeric features having null values with "zero"



num_null_sub_feats = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 

                      'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']



for f in num_null_sub_feats:

    all_data[f] = train[f].fillna(0)
for f in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[f] = all_data[f].fillna(0)



for f in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    all_data[f] = all_data[f].fillna('None')



for f in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[f] = all_data[f].fillna('None')
# Derive new features



all_data['RemodelOn'] = all_data['YearBuilt'] + all_data['YearRemodAdd']

all_data['TotalSF']   = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']

all_data['PorchSF']   = all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch']



all_data['hasPool']      = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['has2ndfloor']  = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasGarage']    = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasBasement']  = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data.shape
final_data = pd.get_dummies(all_data).reset_index(drop=True)

final_data.shape
y = train['SalePrice']

final_data.drop('SalePrice', axis=1, inplace=True)

X = final_data.iloc[:len(y), :]



X_sub = final_data.iloc[len(y):, :]

X.shape, y.shape, X_sub.shape
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
# Use RobustScaler to have robust estimates for center and range of data (used when data has outliers)

# Scale features using statistics that are robust to outliers; 

# this Scaler removes the median and scales the data according to the quantile range (IQR)



ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))

svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003,))
gbr = GradientBoostingRegressor(n_estimators=3000, 

                                learning_rate=0.05, max_depth=4, 

                                max_features='sqrt', min_samples_leaf=15, min_samples_split=10, 

                                loss='huber', random_state=42)  
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
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:squarederror', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                    meta_regressor=xgboost,

                                    use_features_in_secondary=True)
score = cv_rmse(ridge)

print("Ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())



score = cv_rmse(lasso)

print("LASSO: {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())



score = cv_rmse(elasticnet)

print("elastic net: {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())



score = cv_rmse(svr)

print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())



score = cv_rmse(gbr)

print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())



score = cv_rmse(xgboost)

print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()), datetime.now())
print('Start Fitting models..')



print('Ridge')

ridge_model_full_data = ridge.fit(X, y)



print('Lasso')

lasso_model_full_data = lasso.fit(X, y)



print('elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)



print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))



print('Svr')

svr_model_full_data = svr.fit(X, y)



print('lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)



print('GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)



print('xgboost')

xgb_model_full_data = xgboost.fit(X, y)



print('Model fitting complete!')
def blend_models_predict(X):

    return (

            (0.10 * ridge_model_full_data.predict(X)) + \

            (0.05 * lasso_model_full_data.predict(X)) + \

            (0.10 * elastic_model_full_data.predict(X)) + \

            (0.30 * stack_gen_model.predict(np.array(X))) + \

            (0.10 * svr_model_full_data.predict(X)) + \

            (0.10 * lgb_model_full_data.predict(X)) + \

            (0.10 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X))

    )
print('RMSLE score on train data = ', rmsle(y, blend_models_predict(X)))
# Predict on test data set



boxed_predictions = blend_models_predict(X_sub)
# Inverse transform boxcox SalePrice to actual SalePrice values



list_predictions = inv_boxcox(boxed_predictions, lambda_)
# Glance over the SalePrice values



actual_predictions = pd.DataFrame(list_predictions.tolist(), columns=['SalePrice'], index=test['Id'])

actual_predictions
# Save predicted SalePrice to file

actual_predictions.to_csv('/kaggle/working/hpp_predictions.csv')