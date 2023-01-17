# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

from statistics import mode

from scipy.special import boxcox1p



# Machine Learning

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import Lasso, Ridge, RidgeCV, ElasticNet

import xgboost as xgb

import lightgbm as lgb

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from catboost import Pool, CatBoostRegressor, cv



import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Bring training data into the environment

train = pd.read_csv('../input/train.csv')



# Bring test data into the environment

test = pd.read_csv('../input/test.csv')



train.head()
# Let's get confirmation on the dataframe shapes

print("\nThe train data size is: {} ".format(train.shape)) 

print("The test data size is: {} ".format(test.shape))
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



# Now drop the 'Id' colum since it's unnecessary for the prediction process

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
# Checking for outliers in GrLivArea as indicated in dataset documentation

sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=True)

plt.show()
# Removing two very extreme outliers in the bottom right hand corner

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



# Re-check graph

sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=True)

plt.show()
(mu, sigma) = norm.fit(train['SalePrice'])



# 1. Plot Sale Price

sns.distplot(train['SalePrice'] , fit=norm);

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')



# Get the fitted parameters used by the function

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# 2. Plot SalePrice as a QQPlot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# Applying a log(1+x) transformation to SalePrice

train["SalePrice"] = np.log1p(train["SalePrice"])

# 1. Plot Sale Price

sns.distplot(train['SalePrice'] , fit=norm);

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# 2. Plot SalePrice as a QQPlot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# Saving train & test shapes

ntrain = train.shape[0]

ntest = test.shape[0]



# Creating y_train variable

y_train = train.SalePrice.values



# New all encompassing dataset

all_data = pd.concat((train, test)).reset_index(drop=True)



# Dropping the target

all_data.drop(['SalePrice'], axis=1, inplace=True)



# Printing all_data shape

print("all_data size is: {}".format(all_data.shape))
# Getting a missing % count

all_data_missing = (all_data.isnull().sum() / len(all_data)) * 100

all_data_missing = all_data_missing.drop(all_data_missing[all_data_missing == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Percentage':all_data_missing})

missing_data.head(30)
# Visualising missing data

f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Missing Percentage'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
# Initiate correlation matrix

corr = train.corr()

# Set-up mask

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set-up figure

plt.figure(figsize=(14, 8))

# Title

plt.title('Overall Correlation of House Prices', fontsize=18)

# Correlation matrix

sns.heatmap(corr, mask=mask, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})

plt.show()
# All columns where missing values can be replaced with 'None'

for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):

    all_data[col] = all_data[col].fillna('None')

# All columns where missing values can be replaced with 0

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):

    all_data[col] = all_data[col].fillna(0)

# All columns where missing values can be replaced with the mode (most frequently occurring value)

for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Imputing LotFrontage with the median (middle) value

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
all_data_missing = (all_data.isnull().sum() / len(all_data)) * 100

all_data_missing = all_data_missing.drop(all_data_missing[all_data_missing == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio':all_data_missing})

missing_data.head(30)
# Converting those variables which should be categorical, rather than numeric

for col in ('MSSubClass', 'OverallCond', 'YrSold', 'MoSold'):

    all_data[col] = all_data[col].astype(str)

    

all_data.info()
# Applying a log(1+x) transformation to all skewed numeric features

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Compute skewness

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(15)
# Check on number of skewed features above 75% threshold

skewness = skewness[abs(skewness) > 0.75]

print("Total number of features requiring a fix for skewness is: {}".format(skewness.shape[0]))
# Now let's apply the box-cox transformation to correct for skewness

skewed_features = skewness.index

lam = 0.15

for feature in skewed_features:

    all_data[feature] = boxcox1p(all_data[feature], lam)
# Creating a new feature: Total Square Footage

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# Identifying features where a class is over 97% represented

low_var_cat = [col for col in all_data.select_dtypes(exclude=['number']) if 1 - sum(all_data[col] == mode(all_data[col]))/len(all_data) < 0.03]

low_var_cat
# Dropping these columns from both datasets

all_data = all_data.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis=1)
# List of columns to Label Encode

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



# Process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# Check on data shape        

print('Shape all_data: {}'.format(all_data.shape))
# Get dummies

all_data = pd.get_dummies(all_data)



all_data.shape
# Now to return to separate train/test sets for Machine Learning

train = all_data[:ntrain]

test = all_data[ntrain:]
# Set up variables

X_train = train

X_test = test



# Defining two rmse_cv functions

def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10))

    return(rmse)
# Setting up list of alpha's

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30]



# Iterate over alpha's

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

# Plot findings

cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation")

plt.xlabel("Alpha")

plt.ylabel("Rmse")
# 5 looks like the optimal alpha level, so let's fit the Ridge model with this value

model_ridge = Ridge(alpha = 5)
# Setting up list of alpha's

alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001]



# Iterate over alpha's

cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]



# Plot findings

cv_lasso = pd.Series(cv_lasso, index = alphas)

cv_lasso.plot(title = "Validation")

plt.xlabel("Alpha")

plt.ylabel("Rmse")
# Initiating Lasso model

model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005))
# Setting up list of alpha's

alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001]



# Iterate over alpha's

cv_elastic = [rmse_cv(ElasticNet(alpha = alpha)).mean() for alpha in alphas]



# Plot findings

cv_elastic = pd.Series(cv_elastic, index = alphas)

cv_elastic.plot(title = "Validation")

plt.xlabel("Alpha")

plt.ylabel("Rmse")
# Initiating ElasticNet model

model_elastic = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005))
# Setting up list of alpha's

alphas = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]



# Iterate over alpha's

cv_krr = [rmse_cv(KernelRidge(alpha = alpha)).mean() for alpha in alphas]



# Plot findings

cv_krr = pd.Series(cv_krr, index = alphas)

cv_krr.plot(title = "Validation")

plt.xlabel("Alpha")

plt.ylabel("Rmse")
# Initiatiing KernelRidge model

model_krr = make_pipeline(RobustScaler(), KernelRidge(alpha=6, kernel='polynomial', degree=2.65, coef0=6.9))
# Initiating Gradient Boosting Regressor

model_gbr = GradientBoostingRegressor(n_estimators=1200, 

                                      learning_rate=0.05,

                                      max_depth=4, 

                                      max_features='sqrt',

                                      min_samples_leaf=15, 

                                      min_samples_split=10, 

                                      loss='huber',

                                      random_state=5)
# Initiating XGBRegressor

model_xgb = xgb.XGBRegressor(colsample_bytree=0.2,

                             learning_rate=0.06,

                             max_depth=3,

                             n_estimators=1150)
# Initiating LGBMRegressor model

model_lgb = lgb.LGBMRegressor(objective='regression',

                              num_leaves=4,

                              learning_rate=0.05, 

                              n_estimators=1080,

                              max_bin=75, 

                              bagging_fraction=0.80,

                              bagging_freq=5, 

                              feature_fraction=0.232,

                              feature_fraction_seed=9, 

                              bagging_seed=9,

                              min_data_in_leaf=6, 

                              min_sum_hessian_in_leaf=11)
# Initiating CatBoost Regressor model

model_cat = CatBoostRegressor(iterations=2000,

                              learning_rate=0.10,

                              depth=3,

                              l2_leaf_reg=4,

                              border_count=15,

                              loss_function='RMSE',

                              verbose=200)



# Initiating parameters ready for CatBoost's CV function, which I will use below

params = {'iterations':2000,

          'learning_rate':0.10,

          'depth':3,

          'l2_leaf_reg':4,

          'border_count':15,

          'loss_function':'RMSE',

          'verbose':200}
# Fitting all models with rmse_cv function, apart from CatBoost

cv_ridge = rmse_cv(model_ridge).mean()

cv_lasso = rmse_cv(model_lasso).mean()

cv_elastic = rmse_cv(model_elastic).mean()

cv_krr = rmse_cv(model_krr).mean()

cv_gbr = rmse_cv(model_gbr).mean()

cv_xgb = rmse_cv(model_xgb).mean()

cv_lgb = rmse_cv(model_lgb).mean()
# Define pool

pool = Pool(X_train, y_train)



# CV Catboost algorithm

cv_cat = cv(pool=pool, params=params, fold_count=10, shuffle=True)
# Select best model

cv_cat = cv_cat.at[1999, 'train-RMSE-mean']
# Creating a table of results, ranked highest to lowest

results = pd.DataFrame({

    'Model': ['Ridge',

              'Lasso',

              'ElasticNet',

              'Kernel Ridge',

              'Gradient Boosting Regressor',

              'XGBoost Regressor',

              'Light Gradient Boosting Regressor',

              'CatBoost'],

    'Score': [cv_ridge,

              cv_lasso,

              cv_elastic,

              cv_krr,

              cv_gbr,

              cv_xgb,

              cv_lgb,

              cv_cat]})



# Build dataframe of values

result_df = results.sort_values(by='Score', ascending=True).reset_index(drop=True)

result_df.head(8)
# Plotting model performance

f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation='90')

sns.barplot(x=result_df['Model'], y=result_df['Score'])

plt.xlabel('Models', fontsize=15)

plt.ylabel('Model performance', fontsize=15)

plt.ylim(0.10, 0.116)

plt.title('RMSE', fontsize=15)
# Fit and predict all models

model_lasso.fit(X_train, y_train)

lasso_pred = np.expm1(model_lasso.predict(X_test))



model_elastic.fit(X_train, y_train)

elastic_pred = np.expm1(model_elastic.predict(X_test))



model_ridge.fit(X_train, y_train)

ridge_pred = np.expm1(model_ridge.predict(X_test))



model_xgb.fit(X_train, y_train)

xgb_pred = np.expm1(model_xgb.predict(X_test))



model_gbr.fit(X_train, y_train)

gbr_pred = np.expm1(model_gbr.predict(X_test))



model_lgb.fit(X_train, y_train)

lgb_pred = np.expm1(model_lgb.predict(X_test))



model_krr.fit(X_train, y_train)

krr_pred = np.expm1(model_krr.predict(X_test))



model_cat.fit(X_train, y_train)

cat_pred = np.expm1(model_cat.predict(X_test))
# Create stacked model

stacked = (lasso_pred + elastic_pred + ridge_pred + xgb_pred + lgb_pred + krr_pred + gbr_pred) / 7
# Setting up competition submission

sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = stacked

sub.to_csv('house_price_predictions.csv',index=False)