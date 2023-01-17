#Importing Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (12,6)
#Read the data

data = pd.read_csv('../input/avocado-prices/avocado.csv', index_col = 0)
data.shape
#Looking top 5 rows

data.head()
#Checking for null values

data.isnull().sum().sum()
#Checking for duplicated values

data[data.duplicated()]
#Checking the unique values for 'type' column

data.type.unique()
#Checking the unique values for 'region' column

data.region.unique()
#Checking the distribution of price

data['AveragePrice'].hist(bins = 25)
data['Date'] = pd.to_datetime(data['Date'])
#Creating 3 new columns 'Year', 'Month' and 'Day'

data['month'] = data['Date'].apply(lambda x: x.month)
data['day'] = data['Date'].apply(lambda x: x.day)
data.head()
#We will not need anymore 'date' column so we can drop it

data.drop('Date', axis = 1, inplace = True)
data.head()
year_feature = ['year', 'month', 'day']

for feature in year_feature:
    data.groupby(feature)['AveragePrice'].mean().plot.bar()
    plt.title(feature+' vs Price')
    plt.xlabel(feature)
    plt.ylabel('Average Price')
    plt.show()
numerical_feature = [feature for feature in data.columns if data[feature].dtype != 'O' and feature not in year_feature]
data[numerical_feature].head()
discrete_feature = [feature for feature in numerical_feature if len(data[feature].unique()) < 25 and feature not in ['AveragePrice']]
discrete_feature
continues_feature = [feature for feature in numerical_feature if feature not in discrete_feature]
continues_feature
for feature in continues_feature:
    data[feature].hist(bins = 25)
    plt.title(feature)
    plt.xlabel(feature)
    plt.show()
for feature in continues_feature:
    df = data.copy()
    if 0 in df[feature].unique():
        pass
    else:
        df[feature] = np.log(df[feature])
        df[feature].hist(bins = 25)
        plt.title(feature)
        plt.xlabel(feature)
        plt.show()
for feature in continues_feature:
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column = feature)
        plt.title(feature)
        plt.ylabel(feature)
        plt.show()
data.head()
categorical_feature = [feature for feature in data.columns if data[feature].dtype == 'O']
data[categorical_feature].head()
for feature in categorical_feature:
    data.groupby(feature)['AveragePrice'].mean().plot.bar()
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Average Price')
    plt.show()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
data[categorical_feature] = data[categorical_feature].apply(label.fit_transform)
data.head()
feature_scale = [feature for feature in data.columns if feature not in ['AveragePrice']]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
##data[feature_scale] = scaler.fit_transform(data[feature_scale])
scaler.fit(data[feature_scale])
scaler.transform(data[feature_scale])
data = pd.concat([data['AveragePrice'].reset_index(drop = True), pd.DataFrame(scaler.transform(data[feature_scale]), 
                                                                       columns = feature_scale)], axis = 1)
data.head()
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
X = data.drop('AveragePrice', axis = 1)
y = data['AveragePrice']
#feat_sel_model = SelectFromModel(Lasso(alpha = 0.005, random_state = 42))
#feat_sel_model.fit(X, y)
#feat_sel_model.get_support()
#selected_feat = X.columns[feat_sel_model.get_support()]
#X = X[selected_feat]
#Import Libraries

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import math
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
def test_model(model, X_train = X_train, y_train = y_train):
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv = cv, scoring=make_scorer(r2_score))
    return print('Model Score:', score.mean())
linear = LinearRegression()
linear.fit(X_train, y_train)
test_model(linear)
lasso = Lasso()
lasso.fit(X_train, y_train)
test_model(lasso)
ridge = Ridge()
ridge.fit(X_train, y_train)
test_model(ridge)
random = RandomForestRegressor()
random.fit(X_train, y_train)
test_model(random)
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
test_model(xgb)
## RMSE:
linear_pred = linear.predict(X_test)
print('Linear Score:', r2_score(y_test, linear_pred))
print('Linear RMSE:', math.sqrt(mean_squared_error(y_test, linear_pred)))
lasso_pred = lasso.predict(X_test)
print('Lasso Score:', r2_score(y_test, lasso_pred))
print('Lasso RMSE:', math.sqrt(mean_squared_error(y_test, lasso_pred)))
ridge_pred = ridge.predict(X_test)
print('Ridge Score:', r2_score(y_test, ridge_pred))
print('Ridge RMSE:', math.sqrt(mean_squared_error(y_test, ridge_pred)))
random_pred = random.predict(X_test)
print('Random Score:', r2_score(y_test, random_pred))
print('Random RMSE:', math.sqrt(mean_squared_error(y_test, random_pred)))
xgb_pred = xgb.predict(X_test)
print('XGBoost Score:', r2_score(y_test, xgb_pred))
print('XGBoost RMSE:', math.sqrt(mean_squared_error(y_test, xgb_pred)))
real = pd.DataFrame(np.exp(y_test))
linear = pd.DataFrame(np.exp(linear_pred))
random = pd.DataFrame(np.exp(random_pred))
xgboost = pd.DataFrame(np.exp(xgb_pred))
lasso = pd.DataFrame(np.exp(lasso_pred))
ridge = pd.DataFrame(np.exp(ridge_pred))
df = pd.concat([real, linear, random, xgboost, lasso, ridge], axis = 1)
df.columns = ['Real', 'Linear', 'Random', 'XGBoost', 'Lasso', 'Ridge']
df.dropna(inplace = True)
df.head()
n_estimators = [100, 200, 500, 700, 900, 1000, 1100, 1500]
learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
max_depth = [1,3,5,7,15]
min_child_weight = [1,2,3,4,5]
booster = ['gbtree', 'gblinear']
base_score = [0.25, 0.5, 0.75, 1]

hyper_parameter = {
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'gamma': gamma,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,
    'booster': booster,
    'base_score': base_score
}
random_search = RandomizedSearchCV(xgb, param_distributions=hyper_parameter, n_iter=50,
                                  cv= 5, n_jobs=-1, verbose = 3, return_train_score=True,
                                 scoring=make_scorer(r2_score), random_state=42)
random_search.fit(X_train, y_train)
random_search.best_estimator_
xgb = XGBRegressor(base_score=1, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0.0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.25, max_delta_step=0, max_depth=15,
             min_child_weight=5, missing=None, monotone_constraints='()',
             n_estimators=500, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
xgb.fit(X_train, y_train)
test_model(xgb)
xgb_pred = xgb.predict(X_test)
print('XGBoost Score:', r2_score(y_test, xgb_pred))
print('XGBoost RMSE:', math.sqrt(mean_squared_error(y_test, xgb_pred)))
#The best model is XGboost
