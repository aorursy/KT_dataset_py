# Importing main packages and settings

import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

pd.set_option('display.max_columns', 50)
# Loading the training dataset

df_train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])



# Adding feature for yearmonth of purchase

df_train['yearmonth'] = df_train['timestamp'].dt.year*100 + df_train["timestamp"].dt.month



# Adding log price for use as target variable

df_train['log_price_doc'] = np.log1p(df_train['price_doc'].values)
# Displaying all columns 

df_train.head(5)
# Initial dataframe inspection

df_train.info()

df_train.columns
# dataframe descriptive statistics

df_train.describe()
# creating 4 additional dataframes:

# 1. without columns containing NaN data

df_train_filt = df_train.dropna(axis=1)



# 2. without object columns (to be added back later)

df_nonobject = df_train_filt.select_dtypes(exclude=['object', 'datetime64'])

df_nonobject = df_nonobject.drop(['price_doc', 'log_price_doc'], axis=1)



# 3. object features only

df_object =  df_train_filt.select_dtypes(include=['object'])



# 4. target variable

df_target = df_train_filt['price_doc'].reset_index()
# checking the number of columns for both dataframes

df_train_filt.info()

df_nonobject.info()

df_object.info()

df_target.info()
df_obj_dummies = pd.get_dummies(df_object, drop_first=True)

df_obj_target_dummies = pd.concat([df_obj_dummies, df_target], axis=1).drop('index', axis=1)
# Create df of combined object features and targets

df_obj_target = pd.concat([df_object, df_target], axis=1).drop('index', axis=1)

df_obj_target.head(5)
# create dummy variables for object features

df_obj_target_dummies = pd.get_dummies(df_obj_target, drop_first=True)

df_obj_target_dummies.head()
# Plotting distribution of price and log price

fig, ax = plt.subplots(2,2,figsize=(10,10))



plt.subplot(2,1,1)

df_train['price_doc'].plot(kind='hist', bins=100)



plt.subplot(2,1,2)

df_train['log_price_doc'].plot(kind='hist', bins=100, color=['green'])



plt.show()
# Feature selection

# Courtesy of https://www.kaggle.com/sudalairajkumar/sberbank-russian-housing-market...

# .../simple-exploration-notebook-sberbank



xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



y_train = df_target['price_doc']

x_train = df_nonobject.drop(['id'],axis = 1)

dtrain = xgb.DMatrix(x_train,y_train,feature_names = x_train.columns.values)

model = xgb.train(dict(xgb_params,silent=0),dtrain,num_boost_round=100)



fig,ax=plt.subplots(figsize = (12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show;
X = df_train_filt[['full_sq', 'yearmonth', 'metro_min_avto', 'area_m']].values

y = df_train_filt['price_doc'].values
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression(normalize=True)



lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



print(mean_squared_error(y_test, y_pred))

print(r2_score(y_test, y_pred))
# removing warning just for now

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



gbr = GradientBoostingRegressor()



gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)



print(mean_squared_error(y_test, y_pred))

print(r2_score(y_test, y_pred))
X_nonobj = df_nonobject.values

y = df_train_filt['price_doc'].values



X_nonobj_train, X_nonobj_test, y_train, y_test = train_test_split(X_nonobj, y, random_state=0)
gbr = GradientBoostingRegressor()



gbr.fit(X_nonobj_train, y_train)

y_pred = gbr.predict(X_nonobj_test)



print(mean_squared_error(y_test, y_pred))

print(r2_score(y_test, y_pred))
feature_importance = gbr.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5



fig = plt.figure(figsize=(32, 24))

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, df_train_filt.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()



print(sorted_idx)
X_obj = df_obj_target_dummies.drop(['price_doc'], axis=1).values

y_obj = df_obj_target_dummies['price_doc'].values



# Instantiate a ridge regressor: ridge

ridge = Ridge(alpha=0.5, normalize=True)



# Perform 5-fold cross-validation: ridge_cv

ridge_cv = cross_val_score(ridge, X_obj, y_obj, cv=5)



# Print the cross-validated scores

print(ridge_cv)
X_obj = df_obj_target_dummies.drop('price_doc', axis=1).values

y_obj = df_obj_target_dummies['price_doc'].values



# Instantiate a ridge regressor: ridge

ridge = Ridge(alpha=0.5, normalize=True)



# Perform 5-fold cross-validation: ridge_cv

ridge_cv = cross_val_score(ridge, X_obj, y_obj, cv=5)



# Print the cross-validated scores

print(ridge_cv)
print(X.shape)

print(y_train.shape)
print(X_train.shape)



lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)



print(lasso.coef_)

print(lasso.intercept_)



model = SelectFromModel(lasso, threshold='median', prefit=True)

X_train_new = model.transform(X_train)

print(X_train_new.shape)