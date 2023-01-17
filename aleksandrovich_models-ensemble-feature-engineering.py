import sys

import warnings



import lightgbm as lgb

import numpy as np

import pandas as pd

from mlxtend.regressor import StackingCVRegressor

from scipy.stats import skew

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import ElasticNetCV

from sklearn.linear_model import LassoCV

from sklearn.linear_model import RidgeCV

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVR

from xgboost import XGBRegressor





print("Imports have been set")



# Disabling warnings

if not sys.warnoptions:

    warnings.simplefilter("ignore")
# Reading the training/val data and the test data

X = pd.read_csv('../input/train.csv', index_col='Id')

X_test = pd.read_csv('../input/test.csv', index_col='Id')



# Rows before:

rows_before = X.shape[0]

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

rows_after = X.shape[0]

print("\nRows containing NaN in SalePrice were dropped: " + str(rows_before - rows_after))



# Logarithming target variable in order to make distribution better

X['SalePrice'] = np.log1p(X['SalePrice'])



y = X['SalePrice'].reset_index(drop=True)

train_features = X.drop(['SalePrice'], axis=1)



# concatenate the train and the test set as features for tranformation to avoid mismatch

features = pd.concat([train_features, X_test]).reset_index(drop=True)

print('\nFeatures size:', features.shape)
nan_count_table = (features.isnull().sum())

nan_count_table = nan_count_table[nan_count_table > 0].sort_values(ascending=False)

print("\nColums containig NaN: ")

print(nan_count_table)



columns_containig_nan = nan_count_table.index.to_list()

print("\nWhat values they contain: ")

print(features[columns_containig_nan])
for column in columns_containig_nan:



    # populating with 0

    if column in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

                  'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'TotalBsmtSF',

                  'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',

                  'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea']:

        features[column] = features[column].fillna(0)



    # populate with 'None'

    if column in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', "PoolQC", 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

                  'BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu', 'Fence', 'MiscFeature']:

        features[column] = features[column].fillna('None')



    # populate with most frequent value for cateforic

    if column in ['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'RoofStyle',

                  'Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'RoofMatl', 'ExterQual', 'ExterCond',

                  'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition']:

        features[column] = features[column].fillna(features[column].mode()[0])



# MSSubClass: Numeric feature. Identifies the type of dwelling involved in the sale.

#     20  1-STORY 1946 & NEWER ALL STYLES

#     30  1-STORY 1945 & OLDER

#     40  1-STORY W/FINISHED ATTIC ALL AGES

#     45  1-1/2 STORY - UNFINISHED ALL AGES

#     50  1-1/2 STORY FINISHED ALL AGES

#     60  2-STORY 1946 & NEWER

#     70  2-STORY 1945 & OLDER

#     75  2-1/2 STORY ALL AGES

#     80  SPLIT OR MULTI-LEVEL

#     85  SPLIT FOYER

#     90  DUPLEX - ALL STYLES AND AGES

#    120  1-STORY PUD (Planned Unit Development) - 1946 & NEWER

#    150  1-1/2 STORY PUD - ALL AGES

#    160  2-STORY PUD - 1946 & NEWER

#    180  PUD - MULTILEVEL - INCL SPLIT LEV/FOYER

#    190  2 FAMILY CONVERSION - ALL STYLES AND AGES



# Stored as number so converted to string.

features['MSSubClass'] = features['MSSubClass'].apply(str)

features["MSSubClass"] = features["MSSubClass"].fillna("Unknown")

# MSZoning: Identifies the general zoning classification of the sale.

#    A    Agriculture

#    C    Commercial

#    FV   Floating Village Residential

#    I    Industrial

#    RH   Residential High Density

#    RL   Residential Low Density

#    RP   Residential Low Density Park

#    RM   Residential Medium Density



# 'RL' is by far the most common value. So we can fill in missing values with 'RL'

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# LotFrontage: Linear feet of street connected to property

# Groupped by neighborhood and filled in missing value by the median LotFrontage of all the neighborhood

# TODO may be 0 would perform better than median?

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# LotArea: Lot size in square feet.

# Stored as string so converted to int.

features['LotArea'] = features['LotArea'].astype(np.int64)

# Alley: Type of alley access to property

#    Grvl Gravel

#    Pave Paved

#    NA   No alley access



# So. If 'Street' made of 'Pave', so it would be reasonable to assume that 'Alley' might be 'Pave' as well.

features['Alley'] = features['Alley'].fillna('Pave')

# MasVnrArea: Masonry veneer area in square feet

# Stored as string so converted to int.

features['MasVnrArea'] = features['MasVnrArea'].astype(np.int64)
features['YrBltAndRemod'] = features['YearBuilt'] + features['YearRemodAdd']

features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



# If area is not 0 so creating new feature looks reasonable

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



print('Features size:', features.shape)
nan_count_train_table = (features.isnull().sum())

nan_count_train_table = nan_count_train_table[nan_count_train_table > 0].sort_values(ascending=False)

print("\nAre no NaN here now: " + str(nan_count_train_table.size == 0))
numeric_columns = [cname for cname in features.columns if features[cname].dtype in ['int64', 'float64']]

print("\nColumns which are numeric: " + str(len(numeric_columns)) + " out of " + str(features.shape[1]))

print(numeric_columns)



categoric_columns = [cname for cname in features.columns if features[cname].dtype == "object"]

print("\nColumns whice are categoric: " + str(len(categoric_columns)) + " out of " + str(features.shape[1]))

print(categoric_columns)



skewness = features[numeric_columns].apply(lambda x: skew(x))

print(skewness.sort_values(ascending=False))



skewness = skewness[abs(skewness) > 0.5]

features[skewness.index] = np.log1p(features[skewness.index])

print("\nSkewed values: " + str(skewness.index))
# Kind of One-Hot encoding

final_features = pd.get_dummies(features).reset_index(drop=True)



# Spliting the data back to train(X,y) and test(X_sub)

X = final_features.iloc[:len(y), :]

X_test = final_features.iloc[len(X):, :]

print('Features size for train(X,y) and test(X_test):')

print('X', X.shape, 'y', y.shape, 'X_test', X_test.shape)
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



# check maybe 10 kfolds would be better

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



# Kernel Ridge Regression : made robust to outliers

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))



# LASSO Regression : made robust to outliers

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=14, cv=kfolds))



# Elastic Net Regression : made robust to outliers

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))



# Gradient Boosting for regression

gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10,

                                   loss='huber', random_state=5)



# LightGBM regressor.

lgbm = lgb.LGBMRegressor(objective='regression', num_leaves=4,

                         learning_rate=0.01, n_estimators=5000,

                         max_bin=200, bagging_fraction=0.75,

                         bagging_freq=5, feature_fraction=0.2,

                         feature_fraction_seed=7, bagging_seed=7, verbose=-1)



xgb_r = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                     max_depth=3, min_child_weight=0,

                     gamma=0, subsample=0.7,

                     colsample_bytree=0.7,

                     objective='reg:squarederror', nthread=-1,

                     scale_pos_weight=1, seed=27,

                     reg_alpha=0.00006)



stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gboost, xgb_r, lgbm),

                                meta_regressor=xgb_r,

                                use_features_in_secondary=True)



svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
print('\n\nFitting our models ensemble: ')

print('Elasticnet is fitting now...')

elastic_model = elasticnet.fit(X, y)

print('Lasso is fitting now...')

lasso_model = lasso.fit(X, y)

print('Ridge is fitting now...')

ridge_model = ridge.fit(X, y)

print('XGB is fitting now...')

xgb_model = xgb_r.fit(X, y)

print('Gradient Boosting regressor is fitting now...')

gboost_model = gboost.fit(X, y)

print('LGBMRegressor is fitting now...')

lgbm_model = lgbm.fit(X, y)

print('stack_gen is fitting now...')

stack_gen_model = stack_gen.fit(X, y)

print('SVR is fitting now...')

svr_model = svr.fit(X, y)





# model scoring and validation function

# def cv_rmse(the_model, x):

#     return np.sqrt(-cross_val_score(the_model, x, y, scoring="neg_mean_squared_error", cv=kfolds))





# print('\n\nModels evaluating: ')

# score = cv_rmse(ridge_model, X)

# print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#

# score = cv_rmse(lasso_model, X)

# print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#

# score = cv_rmse(elastic_model, X)

# print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#

# score = cv_rmse(xgb_model, X)

# print("xgb_r score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#

# score = cv_rmse(gboost_model, X)

# print("Gradient boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#

# score = cv_rmse(lgbm_model, X)

# print("LGB score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#

# score = cv_rmse(stack_gen_model, X)

# print("Stack gen score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#

# score = cv_rmse(svr_model, X)

# print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

def blend_models(x):

    return ((0.1* elastic_model.predict(x)) +

            (0.1 * lasso_model.predict(x)) +

            (0.05 * ridge_model.predict(x)) +

            (0.1 * svr_model.predict(x)) +

            (0.1 * gboost_model.predict(x)) +

            (0.15 * xgb_model.predict(x)) +

            (0.1 * lgbm_model.predict(x)) +

            (0.3 * stack_gen_model.predict(np.array(x))))

            



def rmsle(y_actual, y_pred):

    return np.sqrt(mean_squared_error(y_actual, y_pred))



# RMSLE score on train data:

# 0.07450991123905183

print('\nRMSLE score on train data:')

print(rmsle(y, blend_models(X)))
submission = pd.read_csv("../input/sample_submission.csv")

submission.iloc[:, 1] = np.expm1(blend_models(X_test))

submission.to_csv("submission.csv", index=False)



print("Submission file is formed")