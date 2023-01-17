import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNet

import xgboost as xgb

from sklearn.ensemble import BaggingRegressor

from sklearn.model_selection import cross_val_score



%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
print("Feature difference btwn datasets",list(set(df_train.columns) - set(df_test.columns)))
# Missing values

df_train.isnull().sum().sort_values(ascending=False)
# Distribution of target variable before and after log transformation

fig, (ax1, ax2) = plt.subplots(ncols=2)

sns.distplot(df_train.SalePrice, ax=ax1)

sns.distplot(np.log(df_train.SalePrice), ax=ax2)
## Feature scaling

# Numerical features with a mean=0 and stdev=1 

def scale_train_features(df):

    for col_var in df.select_dtypes(include=[np.number]).drop('SalePrice', 1).columns:

        df[col_var] = StandardScaler().fit_transform(df[col_var])

    return df

def scale_test_features(df):

    for col_var in df.select_dtypes(include=[np.number]).columns:

        df[col_var] = StandardScaler().transform(df[col_var])

    return df



# numerical_features = df_train.select_dtypes(include=[np.number]).drop(['SalePrice'], axis=1).columns

# df_train.loc[:, numerical_features] = StandardScaler().fit_transform(df_train.loc[:, numerical_features])

# df_test.loc[:, numerical_features] = StandardScaler().transform(df_test.loc[:, numerical_features])



## Standardization/ Normal distribution

# Log transform variables that have a skew greater than 0.8 

# Standardize to follow Gaussian dist of mean=0 and stdev=1

# Note: Normalizing is used to get values btwn 0 and 1

def standardize_values(df):

    for col_var in df.select_dtypes(include=[np.number]):

        if df[col_var].skew(skipna=True) > 0.8:

            df[col_var] = np.log1p(df[col_var])

        else:

            pass

    return df



## Convert categorical variables to numerical

def convert_categoricals(df):

    for cat_col in df.select_dtypes(include=['object']):

        df[cat_col] = pd.factorize(df[cat_col])[0]

    return df



## Fill NaNs with mean()

def fill_nans(df):

    return df.fillna(df.mean())
## Transform data for models

train = fill_nans(convert_categoricals(standardize_values(df_train)))

test = fill_nans(convert_categoricals(standardize_values(df_test)))



# Create matrices

X_train = train.iloc[:,:-1].drop('Id', 1)

print(X_train.shape)

y_train = train.iloc[:,-1]

print(y_train.shape)

X_test = test.drop('Id', 1)

print(X_test.shape)
## Regularization used to handle collinearity, filter out noise from data, and prevent overfitting.

# Regularization introduces additional information, bias, to penalize extreme parameter weights.

# 'alpha' parameter contols the degree of sparsity of coefficiients estimated

# as alpha increases, the model complexity is reduced however

# high values reduce overfitting, values nearing 5 can lead to underfitting



## RMSE vs MAE 

# Squared error penalizes large deviations more and produces larger errors (emphasizes extremes)

# MAE causes the stdevs to average out and is able to handle outliers since it assigns equal weight

# https://www.quora.com/What-is-the-difference-between-squared-error-and-absolute-error
## RidgeCV

# L2 adds the squared sum of weights to cost function

ridge = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

np.sqrt(-cross_val_score(ridge, X_train, y_train, cv=10, scoring='neg_mean_squared_error')).mean()
## LassoCV

# L1 adds the sum of the sbsolute value of the weights which yields sparse feature fectors that may be zero

lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

np.sqrt(-cross_val_score(lasso, X_train, y_train, cv=10, scoring='neg_mean_squared_error')).mean()
# L1 Feature Importance

lasso.fit(X_train, y_train)

lasso_feats = pd.Series(lasso.coef_, index=X_train.columns).sort_values()

lasso_feats = pd.concat([lasso_feats.head(5), lasso_feats.tail(5)])

lasso_feats.plot(kind='barh')
## ElasticNet

# L1 penalty to generate sparsity and L2 to overcome limitations of Lasso, such as number of variables

elastic = ElasticNet(alpha=0.001)

np.sqrt(-cross_val_score(elastic, X_train, y_train, cv=10, scoring='neg_mean_squared_error')).mean()
## Predictions with ElasticNet (lowest RMSE)

# elastic.fit(X_train, y_train)

# elastic_pred = np.exp(elastic.predict(X_test))

# df_elastic = pd.DataFrame(elastic_pred, index=df_test['Id'], columns=['SalePrice'])

# df_elastic.to_csv('/Users/dominicdebiaso/Desktop/kaggle_house_prices_elasticnet.csv')



# Perform Bootstrap Aggregation on ElasticNet

# Fit base regressors on a random subset of original and aggregate individual predictions

# Redduces variace by introducing randomization

# Bootstrap: drawn with replacement

bagged = BaggingRegressor(base_estimator=elastic, n_estimators=100)

bagged.fit(X_train, y_train)

bagged_pred = np.exp(bagged.predict(X_test))

df_bagged = pd.DataFrame(bagged_pred, index=df_test['Id'], columns=['SalePrice'])
# Determine optimal 'n_estimators'

dtrain = xgb.DMatrix(X_train, y_train)

dtest = xgb.DMatrix(X_test)



param = {'eta': 0.01,

         'max_depth': 5,

         'subsample':0.7,

         'objective':'reg:linear'}

xgb_cv = xgb.cv(param, dtrain, num_boost_round=3000, nfold=5, metrics=['rmse'], early_stopping_rounds=100)

best_nrounds = xgb_cv.shape[0]



bst = xgb.train(param, dtrain, num_boost_round=best_nrounds)

preds = np.exp(bst.predict(dtest))

df_xgb = pd.DataFrame(preds, index=df_test['Id'], columns=['SalePrice'])
# Scikit XGBRegressor to ensemble with Bagged ElasticNet

param = {'learning_rate': 0.01,

         'max_depth': 5,

         'subsample':0.7,

         'n_estimators': best_nrounds,

         'objective':'reg:linear'}

xgb_reg = xgb.XGBRegressor()

xgb_reg.fit(X_train, y_train)

xgb_reg_preds = np.exp(xgb_reg.predict(X_test))
# Create ensemble model by averaging

ensemble_pred = (0.6*bagged_pred) + (0.4*xgb_reg_preds)

df_ensemble = pd.DataFrame(ensemble_pred, index=df_test['Id'], columns=['SalePrice'])
# Source: https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models/notebook