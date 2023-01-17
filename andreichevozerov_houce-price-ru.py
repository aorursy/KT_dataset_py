
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.stats import norm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

display(train_df.head())
train_df.columns = [x.lower() for x in train_df.columns]
test_df.columns = [x.lower() for x in test_df.columns]

display(train_df.head())
train_df['saleprice'] = np.log(train_df['saleprice'])
sns.distplot(train_df['saleprice'], fit=norm)
anomalies_list = ['lotfrontage', 'lotarea', 'masvnrarea', 'bsmtfinsf1', 'bsmtunfsf', 'totalbsmtsf', '1stflrsf', 'grlivarea']

for col in anomalies_list:
    q3 = train_df[col].describe()[6]
    q1 = train_df[col].describe()[4]
    interquartile_range = (q3 - q1) * 1.5
    train_df = train_df[(train_df[col] > q1 - interquartile_range) & (train_df[col] < q3 + interquartile_range)]

train_df = train_df.query('miscval < 8000')

train_df.shape
y_train = train_df['saleprice'].reset_index(drop=True)
train_df = train_df.drop(['saleprice'], axis=1)

train_df.drop(['id'], axis=1, inplace=True)
test_df.drop(['id'], axis=1, inplace=True)

all_features = pd.concat((train_df,test_df)).reset_index(drop=True)
all_features.shape
all_features['lotfrontage'] = all_features['lotfrontage'].fillna(all_features['lotfrontage'].median())

na_dict = {
    'masvnrtype': 'None',
    'masvnrarea': 0.0,
    'bsmtqual': 'None',
    'bsmtcond': 'None',
    'bsmtexposure': 'None',
    'bsmtfintype1': 'None',
    'bsmtfintype2': 'None',
    'electrical': 'SBrkr',
    'fireplacequ': 'None',
    'garagetype': 'None',
    'garageyrblt': 'None',
    'garagefinish': 'None',
    'garagequal': 'None',
    'garagecond': 'None',
    'poolqc': 'None',
    'fence': 'None',
    'miscfeature': 'None',
    'alley': 0
}

for col in all_features.columns[1::1]:
    try:
        all_features[col] = all_features[col].fillna(na_dict[col])
    except:
        continue

all_features = all_features.dropna()
print(all_features.info())
all_features = pd.get_dummies(all_features)

all_features.shape
X = all_features.iloc[:len(y_train), :]
X_test = all_features.iloc[len(y_train):, :]
X.shape, y_train.shape, X_test.shape
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.95:
        overfit.append(i)

overfit = list(overfit)

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

kfolds = KFold(n_splits=16, shuffle=True, random_state=42)

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y_train, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
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
print('cancel')
def blend_predictions(X):
    return ((0.12  * elastic_model.predict(X)) + \
            (0.12 * lasso_model.predict(X)) + \
            (0.12 * ridge_model.predict(X)) + \
            (0.22 * lgb_model.predict(X)) + \
            (0.1 * svr_model.predict(X)) + \
            (0.32 * stack_gen_model.predict(np.array(X))))
blended_score = rmsle(y_train, blend_predictions(X))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.loc[:, 'SalePrice'] = np.floor(np.expm1(blend_predictions(X_test)))

#submission.to_csv("submission.csv", index=False)
display(submission)
print(np.floor(np.expm1(blend_predictions(X_test))))
