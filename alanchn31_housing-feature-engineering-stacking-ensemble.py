import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_rows', 85)



import os

print(os.listdir("../input"))



from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from lightgbm import LGBMRegressor
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

full_data = [df_train, df_test]
df_train.describe()
df_train.head()
sns.distplot(df_train['SalePrice'], bins = 20, hist_kws=dict(edgecolor="k", linewidth=2))
sns.distplot(np.log(df_train['SalePrice']), bins = 20, hist_kws=dict(edgecolor="k", linewidth=2))
df_train['LSalePrice'] = np.log(df_train['SalePrice'])
import statsmodels.api as sm
fig = sm.qqplot(df_train['LSalePrice'],fit=True, line='45')
from scipy import stats

m = stats.trim_mean(df_train['LSalePrice'], 0.1)

print("With 10% clipped on both sides, trimmed mean: {}".format(m))

print("Sample mean: {}".format(np.mean(df_train['LSalePrice'])))
df_train.dtypes
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)



corr = df_train.corr()

ax = sns.heatmap(corr, vmin=0.5, cmap='coolwarm', ax=ax)
sns.pairplot(df_train[["SalePrice","OverallQual","OverallCond","YearBuilt","TotalBsmtSF","BsmtUnfSF","GrLivArea","FullBath","GarageCars","GarageArea"]])
df_train.isnull().sum()
df_test.isnull().sum()
len([ 'LotFrontage', 'Alley', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'])
len(['MSZoning', 'LotFrontage','Alley','Utilities','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea', 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath','KitchenQual','Functional','FireplaceQu', 'GarageType', 'GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond'])
df_train['MSSubClass'][df_train['LotFrontage'].isnull()].unique()
a4_dims = (11.7, 6.27)

fig, ax = plt.subplots(figsize=a4_dims)

fig2, ax2 = plt.subplots(figsize=a4_dims)

sns.boxplot(x="MSSubClass",y="LotFrontage",data=df_train,ax=ax).set_title("Training set")

sns.boxplot(x="MSSubClass",y="LotFrontage",data=df_test,ax=ax2).set_title("Test set")
for dataset in full_data:

    dataset['LotFrontage'] = dataset.groupby('MSSubClass')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df_test.groupby('MSSubClass')['LotFrontage'].median()
df_test['LotFrontage'][df_test['MSSubClass']==150] = 38
missin_cats = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature','MasVnrType']

df_train.update(df_train[missin_cats].fillna("NA"))

df_test.update(df_test[missin_cats].fillna("NA"))
df_test[df_test['MSZoning'].isnull()]
def aggregate(rows,columns,df):

    column_keys = df[columns].unique()

    row_keys = df[rows].unique()



    agg = { key : [ len(df[(df[rows]==value) & (df[columns]==key)]) for value in row_keys]

               for key in column_keys }



    aggdf = pd.DataFrame(agg,index = row_keys)

    aggdf.index.rename(rows,inplace=True)



    return aggdf



a4_dims = (11.7, 6.27)

fig, ax = plt.subplots(figsize=a4_dims)

aggregate('MSSubClass','MSZoning',df_test).plot(kind='bar',stacked=True, ax=ax)
aggregate('MSSubClass','MSZoning',df_test)
df_test['MSZoning'][(df_test['MSZoning'].isnull()) & (df_test['MSSubClass']==20) ] = "RL"
df_test['MSZoning'][(df_test['MSZoning'].isnull()) & ( (df_test['MSSubClass']==30) | (df_test['MSSubClass']==70) ) ] = "RM"
for col in ['Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF' ,'TotalBsmtSF', 'KitchenQual', 'Functional','GarageCars', 'GarageArea', 'SaleType','BsmtFullBath','BsmtHalfBath']:

    df_test[col].fillna(df_test[col].mode()[0],inplace=True)
df_test.isnull().sum()
df_train.isnull().sum()
df_train['Electrical'].fillna(df_train['Electrical'].mode()[0],inplace=True)
for data_set in full_data:

    data_set.drop(['GarageYrBlt'], axis=1,inplace=True)
a4_dims = (11.7, 6.27)

fig, ax = plt.subplots(figsize=a4_dims)

fig2, ax2 = plt.subplots(figsize=a4_dims)

sns.boxplot(x="MasVnrType",y="MasVnrArea",data=df_train,ax=ax).set_title("Training set")

sns.boxplot(x="MasVnrType",y="MasVnrArea",data=df_test,ax=ax2).set_title("Test set")
df_train[df_train['MasVnrArea'].isnull()]
for dataset in full_data:

    dataset['MasVnrArea'] = dataset.groupby('MasVnrType')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))
df_train['MasVnrArea'][df_train['MasVnrArea'].isnull()] = 0
sum(df_train.isnull().sum())
sum(df_test.isnull().sum())
df_train['MSSubClass'].unique()
df_test['MSSubClass'].unique()
df_train['train'] = 1

df_test['train'] = 0
combined_dataset = pd.concat([df_train,df_test])
categories={} # contains all the levels in those feature columns

categorical_feature_names =   [ 'MSSubClass' , 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 

                               'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 

                               'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 

                               'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'MiscFeature', 'SaleType', 

                               'SaleCondition', 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',

                               'ExterQual', 'ExterCond', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

                               'BsmtFinType2', 'HeatingQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 

                               'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces' , 

                               'FireplaceQu',  'BsmtQual', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 

                               'Fence',]



for f in categorical_feature_names:

    # to make sure the type is indeed category

    combined_dataset[f] = combined_dataset[f].astype('category')

    categories[f] = combined_dataset[f].cat.categories



new_combined_dataset = pd.get_dummies(combined_dataset,columns=categorical_feature_names,drop_first=True)

df_newtrain = new_combined_dataset[new_combined_dataset['train'] == 1]

df_newtest = new_combined_dataset[new_combined_dataset['train'] == 0]

df_newtrain.drop(["train"],axis=1,inplace=True)

df_newtest.drop(["train","LSalePrice","SalePrice"],axis=1,inplace=True)
df_newtrain.shape
df_newtest.shape
new_fulldata = [df_newtrain, df_newtest]

for dataset in new_fulldata:

    for col in ['YearBuilt','YearRemodAdd']:

        dataset['HouseAge'] = 2018 - dataset['YearBuilt']

        dataset['RemodAge'] = 2018 - dataset['YearRemodAdd']

        dataset.drop(['YearBuilt','YearRemodAdd'],axis=1)
lambgrid = np.linspace(0.01, 100, num=1000,endpoint=True)
y_train = df_train['LSalePrice']
train_id = df_newtrain['Id']

test_id = df_newtest['Id']
X_train = df_newtrain.drop(['SalePrice','LSalePrice','Id'],axis=1)
df_newtest.drop(['Id'],axis=1,inplace=True)
#Ridge Regression:

from sklearn.linear_model import RidgeCV

# clf = RidgeCV(alphas=lambgrid, cv=5)

clf = make_pipeline(RobustScaler(),RidgeCV(alphas=lambgrid, cv=5))
from sklearn.linear_model import Ridge

coefs = []

for a in lambgrid:

    ridge = Ridge(alpha=a, fit_intercept=False)

    ridge.fit(X_train, y_train)

    coefs.append(ridge.coef_)
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

ax.plot(lambgrid, coefs)

ax.set_xscale('log')

ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Ridge coefficients as a function of the regularization')

plt.axis('tight')

plt.show()
clf.fit(X_train,y_train)
preds_ridge = clf.predict(df_newtest)
from sklearn.model_selection import cross_val_score

print('Ridge Regression Cross Validation Score: %s' % (

                      cross_val_score(clf, X_train, y_train,scoring='neg_mean_squared_error').mean()))
#LASSO:

from sklearn.linear_model import LassoCV

# lasso_clf = LassoCV(alphas=lambgrid, cv=5)

lasso_clf = make_pipeline(RobustScaler(), LassoCV(alphas=lambgrid, cv=5))
lasso_clf.fit(X_train,y_train)
preds_lasso = lasso_clf.predict(df_newtest)
# lasso_alpha = lasso_clf.alpha_

# from sklearn.linear_model import Lasso



print('Lasso Cross Validation Score: %s' % (

                      cross_val_score(lasso_clf, X_train, y_train,scoring='neg_mean_squared_error').mean()))
#ElasticNet:

from sklearn.linear_model import ElasticNetCV

# elastic_clf = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],alphas=lambgrid, cv=5)

elastic_clf = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],alphas=lambgrid, cv=5))
elastic_clf.fit(X_train,y_train)
preds_elastic = elastic_clf.predict(df_newtest)
# elastic_alpha = elastic_clf.alpha_

# from sklearn.linear_model import ElasticNet



print('ElasticNet Cross Validation Score: %s' % (

                      cross_val_score(elastic_clf, X_train, y_train,scoring='neg_mean_squared_error').mean()))
import xgboost as xgb

xgtrain = xgb.DMatrix(X_train,label=y_train)

xgb_clf = xgb.XGBRegressor(n_estimators=1000, learning_rate=.03, max_depth=3, max_features=.04, min_samples_split=4,

                           min_samples_leaf=3, loss='huber', subsample=1.0, random_state=0)

xgb_param = xgb_clf.get_xgb_params()
print('XGB Regression Cross Validation Score: %s' % (

                      cross_val_score(xgb_clf, X_train, y_train,scoring='neg_mean_squared_error').mean()))
xgb_clf.fit(X_train,y_train)
preds_xgb = xgb_clf.predict(df_newtest)
# Fit Light Gradient Boosting:

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

                         verbose=-1)



lightgbm.fit(X_train,y_train)
preds_lgb = lightgbm.predict(df_newtest)
## Stacking

from mlxtend.regressor import StackingCVRegressor





ridge = make_pipeline(RobustScaler(),RidgeCV(alphas=lambgrid, cv=5))

lasso = make_pipeline(RobustScaler(), LassoCV(alphas=lambgrid, cv=5))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],alphas=lambgrid, cv=5))

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

xgboost = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)





stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)

stack_gen.fit(np.array(X_train), np.array(y_train))

stacked_preds = stack_gen.predict(np.array(df_newtest))
## Blending:



def blend_models_predict(preds_ridge, preds_xgb, preds_lgb):

    return (np.exp(preds_ridge) + np.exp(preds_xgb) + np.exp(preds_lgb))/3
submission = pd.DataFrame({'Id': test_id, 'SalePrice': np.exp(preds_xgb)})
# Stack Ridge Regression and XGBoost Regression Predictions

blend_preds = blend_models_predict(preds_ridge, preds_xgb, preds_lgb)

blend_submission = pd.DataFrame({'Id': test_id, 'SalePrice': blend_preds})

stacked_submission = pd.DataFrame({'Id': test_id, 'SalePrice': np.exp(stacked_preds)})
# Save Stacked and XGBoost Regression Predictions

submission.to_csv('Submission.csv',index=False)

stacked_submission.to_csv('Stacked_Submission.csv', index=False)

blend_submission.to_csv('Blend_Submission.csv', index=False)