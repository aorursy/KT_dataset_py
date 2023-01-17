import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.preprocessing import StandardScaler

from scipy import stats

%matplotlib inline



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train['SalePrice'].describe()
print(train['SalePrice'].skew())

print(train['SalePrice'].kurt())
sns.distplot(train['SalePrice'], fit=norm)

#create new figure

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log(train['SalePrice'])

sns.distplot(train['SalePrice'], fit=norm)



#create new figure

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
fig, ax = plt.subplots(figsize=(12,9))

sns.heatmap(train.corr(),square=True);
cols = train.corr().nlargest(10, 'SalePrice').index

plt.subplots(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(train[cols].corr(),square=True, annot=True);
sns.pairplot(train[cols]);
var = 'GrLivArea'

plt.scatter(x=train[var], y=train['SalePrice']);
train[train['GrLivArea'] > 4500].index
var = 'GarageArea'

plt.scatter(x=train[var], y=train['SalePrice']);
train[train['GarageArea'] > 1220].index
var = 'TotalBsmtSF'

plt.scatter(x=train[var], y=train['SalePrice']);
train[train['TotalBsmtSF'] > 5000].index
var = '1stFlrSF'

plt.scatter(x=train[var], y=train['SalePrice']);
train[train['TotalBsmtSF'] > 4000].index
var = 'OverallQual'

plt.subplots(figsize=(10,6))

sns.boxplot(x=train[var], y=train['SalePrice']);
var = 'YearBuilt'

plt.subplots(figsize=(20,10))

sns.boxplot(x=train[var], y=train['SalePrice']);
train = train.drop([523, 581, 1061, 1190, 1298])
train.reset_index(drop=True, inplace=True)
y_train = train['SalePrice'].reset_index(drop=True)

train = train.drop(['SalePrice'], axis=1)
#Delete ID

train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
#concatenate train and test

all_features = pd.concat((train,test)).reset_index(drop=True)

all_features.shape
# missing values

total = all_features.isnull().sum().sort_values(ascending=False)

percent = (all_features.isnull().sum()/all_features.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])

missing_data.head(40)
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)

all_features['YrSold'] = all_features['YrSold'].apply(str)

all_features['MoSold'] = all_features['MoSold'].apply(str)
# Some features have only a few missing value. Fill up using most common value

common_vars = ['Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual']

for var in common_vars:

    all_features[var] = all_features[var].fillna(all_features[var].mode()[0])



all_features['MSZoning'] = all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# data description says NA means typical

all_features['Functional'] = all_features['Functional'].fillna('Typ')
col_str = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',

            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"

           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']

for col in col_str:

    all_features[col] = all_features[col].fillna('None')
col_num = ['GarageYrBlt','GarageArea','GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF']

for col in col_num:

    all_features[col] = all_features[col].fillna(0)

    

all_features['LotFrontage'] = all_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_features.isnull().sum().sort_values(ascending=False).head(5)
num_features = all_features.select_dtypes(exclude='object').columns
# Create box plots for all numeric features

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 12))

ax.set_xscale("log")

ax = sns.boxplot(data=all_features[num_features], orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
# Find skewed numerical features

skewness = all_features[num_features].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skewness = skewness[abs(skewness) > 0.5]



print("There are {} numerical features with Skew > 0.5 :".format(high_skewness.shape[0]))

high_skewness.sort_values(ascending=False)
high_skewness.index
from scipy.special import boxcox1p

skewed_features = high_skewness.index

for feat in skewed_features:

    all_features[feat] = boxcox1p(all_features[feat], boxcox_normmax(all_features[feat] + 1))
new_skewness = all_features[num_features].apply(lambda x: skew(x)).sort_values(ascending=False)

new_high_skewness = new_skewness[abs(new_skewness) > 0.5]

print("There are {} skewed numerical features after Box Cox transform".format(new_high_skewness.shape[0]))

print("Mean skewnees: {}".format(np.mean(new_high_skewness)))

new_high_skewness.sort_values(ascending=False)
#  Adding total sqfootage feature 

all_features['TotalSF']=all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']

#  Adding total bathrooms feature

all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +

                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))

#  Adding total porch sqfootage feature

all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +

                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +

                              all_features['WoodDeckSF'])



all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)





# Not normaly distributed can not be normalised and has no central tendecy

all_features = all_features.drop(['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF'], axis=1)
all_features = pd.get_dummies(all_features).reset_index(drop=True)

all_features.shape
X = all_features.iloc[:len(y_train), :]

X_test = all_features.iloc[len(y_train):, :]

X.shape, y_train.shape, X_test.shape
# Removes colums where the threshold of zero's is (> 99.95), means has only zero values 

overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.95:

        overfit.append(i)



overfit = list(overfit)

overfit.append('MSZoning_C (all)')



X = X.drop(overfit, axis=1).copy()

X_test = X_test.drop(overfit, axis=1).copy()



print(X.shape,y_train.shape,X_test.shape)
from datetime import datetime

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

from sklearn.linear_model import LinearRegression



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
# setup cross validation folds

kfolds = KFold(n_splits=16, shuffle=True, random_state=42)



# define error metrics

def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y_train, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
# LightGBM regressor

lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=4,

                       learning_rate=0.01, 

                       n_estimators=9000,

                       max_bin=200, 

                       bagging_fraction=0.75,

                       bagging_freq=5, 

                       bagging_seed=7,

                       feature_fraction=0.2,

                       feature_fraction_seed=7,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)



# XGBoost Regressor

"""xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=42)"""



"""gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)  

"""



# setup models hyperparameters using a pipline

# This is a range of values that the model considers each time in runs a CV

ridge_alpha = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

lasso_alpha = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



elastic_alpha = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



# Ridge Regression: robust to outliers using RobustScaler

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alpha, cv=kfolds))



# Lasso Regression: 

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 

                    alphas=lasso_alpha,random_state=42, cv=kfolds))



# Elastic Net Regression:

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 

                         alphas=elastic_alpha, cv=kfolds, l1_ratio=e_l1ratio))



# Support Vector Regression

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))



# Stack up all the models above

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, svr, lightgbm),

                                meta_regressor=elasticnet,

                                use_features_in_secondary=True)
# Store scores of each model

scores = {}



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lightgbm'] = (score.mean(), score.std())
score = cv_rmse(ridge)

print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['ridge'] = (score.mean(), score.std())
score = cv_rmse(lasso)

print("lasso: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lasso'] = (score.mean(), score.std())
score = cv_rmse(elasticnet)

print("elasticnet: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['elasticnet'] = (score.mean(), score.std())
score = cv_rmse(svr)

print("svr: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['svr'] = (score.mean(), score.std())
"""score = cv_rmse(gbr)

print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['gbr'] = (score.mean(), score.std())"""
print('----START Fit----',datetime.now())

print('Elasticnet')

elastic_model = elasticnet.fit(X, y_train)

print('Lasso')

lasso_model = lasso.fit(X, y_train)

print('Ridge')

ridge_model = ridge.fit(X, y_train)

print('lightgbm')

lgb_model = lightgbm.fit(X, y_train)

print('svr')

svr_model = svr.fit(X, y_train)



print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y_train))
def blend_predictions(X):

    return ((0.12  * elastic_model.predict(X)) + \

            (0.12 * lasso_model.predict(X)) + \

            (0.12 * ridge_model.predict(X)) + \

            (0.22 * lgb_model.predict(X)) + \

            (0.1 * svr_model.predict(X)) + \

            (0.32 * stack_gen_model.predict(np.array(X))))
# Get final precitions from the blended model

blended_score = rmsle(y_train, blend_predictions(X))

scores['blended'] = (blended_score, 0)

print('RMSLE score on train data:')

print(blended_score)
# Plot the predictions for each model

sns.set_style("white")

fig = plt.figure(figsize=(24, 12))



ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])

for i, score in enumerate(scores.values()):

    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')



plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)

plt.xlabel('Model', size=20, labelpad=12.5)

plt.tick_params(axis='x', labelsize=13.5)

plt.tick_params(axis='y', labelsize=12.5)



plt.title('Scores of Models', size=20)



plt.show()
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_predictions(X_test)))
submission.to_csv("submission.csv", index=False)