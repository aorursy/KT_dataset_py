import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (15,8)
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()
data.shape
data.describe()
data.isnull().sum() / len(data) *100
data.info()
data.drop(['id', 'name', 'host_id', 'host_name'], axis = 1, inplace = True)
data.head()
missing_values = [feature for feature in data.columns if data[feature].isnull().sum() > 0]

for feature in missing_values:
    print('{}: {}% missing values'.format(feature, np.round(data[feature].isnull().mean(), 4)))
categorical_nan = [feature for feature in missing_values if data[feature].dtype == 'O']
for feature in categorical_nan:
    print('{}: {}% missing values'.format(feature, np.round(data[feature].isnull().mean(), 4)))
data['last_review'].fillna(data['last_review'].mode()[0], inplace = True)
numerical_nan = [feature for feature in missing_values if data[feature].dtype != 'O']
for feature in numerical_nan:
    print('{}: {}% missing values'.format(feature, np.round(data[feature].isnull().mean(), 4)))
data['reviews_per_month'].fillna(data['reviews_per_month'].median(), inplace = True)
data.isnull().sum().sum()
numerical_feature = [feature for feature in data.columns if data[feature].dtype != 'O']
data[numerical_feature].head()
discrete_feature = [feature for feature in numerical_feature if len(data[feature].unique()) < 25 and feature not in ['price']]
discrete_feature
continues_feature = [feature for feature in numerical_feature if feature not in discrete_feature and feature not in ['longitude']]
list(continues_feature)
for feature in continues_feature:
    df = data.copy()
    df[feature].hist(bins = 25)
    plt.title('Distribution of '+feature)
    plt.xlabel(feature)
    plt.ylabel('Distribution')
    plt.show()
for feature in continues_feature:
    df = data.copy()
    if 0 in df[feature].unique():
        pass
    else:
        df[feature] = np.log(df[feature])
        df[feature].hist(bins = 25)
        plt.title('Distribution of '+feature)
        plt.xlabel(feature)
        plt.ylabel('Distribution')
        plt.show()
categorical_feature = [feature for feature in data.columns if data[feature].dtype == 'O']
data[categorical_feature].head()
for feature in categorical_feature:
    df = data.copy()
    df.groupby(feature)['price'].median().plot.bar()
    plt.title(feature+' vs price per night')
    plt.xlabel(feature)
    plt.ylabel('price -->')
    plt.show()
data.shape
data = data[~(data['price'] == 0)]
data.shape
data.head()
data.boxplot(column = 'price')
q1, q3 = np.percentile(data['price'], [25, 75])
print(q1, '<-->', q3)
iqr = q3 - q1
print('Interquartile:', iqr)
lower = q1 - (1.5 * iqr)
upper = q3 + (1.5 * iqr)
data[data['price'] < lower]
data = data[data['price'] < upper]
data.shape
data.boxplot(column = 'price')
data.head()
data['last_review'] = pd.to_datetime(data['last_review'])
data['year_review'] = data['last_review'].apply(lambda x: x.year)
data['month_review'] = data['last_review'].apply(lambda x: x.month)
data['day_review'] = data['last_review'].apply(lambda x: x.day)
data.drop('last_review', axis = 1, inplace = True)
data.head()
for feature in continues_feature:
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data[feature].hist(bins = 25)
        plt.title(feature)
        plt.xlabel(feature)
        plt.show()
data.head()
categorical_feature = [feature for feature in data.columns if data[feature].dtype == 'O']
data[categorical_feature].head()
for feature in categorical_feature:
    temp = data[feature].value_counts() / len(data)*100
    temp = temp[temp > 1].index
    data[feature] = np.where(data[feature].isin(temp), data[feature], 'Rare_var')
data.head(10)
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
data[categorical_feature] = data[categorical_feature].apply(label.fit_transform)
data.head()
feature_scaler = [feature for feature in data.columns if feature not in ['price']]
data[feature_scaler].head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(data[feature_scaler])
scaler.transform(data[feature_scaler])
data = pd.concat([data['price'].reset_index(drop = True), pd.DataFrame(scaler.transform(data[feature_scaler]),
                columns = feature_scaler)], axis = 1)
data.head()
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
X = data.drop('price', axis = 1)
y = data['price']
feature_sel_model = SelectFromModel(Lasso(alpha = 0.005, random_state = 42))
feature_sel_model.fit(X, y)
feature_sel_model.get_support()
selected_feat = X.columns[feature_sel_model.get_support()]
X = X[selected_feat]
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
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
linear.fit(X, y)
test_model(linear)
linear_pred = linear.predict(X_test)
print('Linear Score:', r2_score(y_test, linear_pred))
print('Linear Error:', mean_squared_error(y_test, linear_pred))
random = RandomForestRegressor()
random.fit(X, y)
test_model(random)
random_pred = random.predict(X_test)
print('Random Score:', r2_score(y_test, random_pred))
print('Random Error:', mean_squared_error(y_test, random_pred))
xgb = XGBRegressor()
xgb.fit(X, y)
test_model(xgb)
xgb_pred = xgb.predict(X_test)
print('XGBoost Score:', r2_score(y_test, xgb_pred))
print('XGBoost Error:', mean_squared_error(y_test, xgb_pred))
np.exp(y_test)
np.exp(random_pred)
np.exp(xgb_pred)
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
xgb = XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0.2, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.05, max_delta_step=0, max_depth=7,
             min_child_weight=4, missing=None, monotone_constraints='()',
             n_estimators=700, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
xgb.fit(X, y)
test_model(xgb)
xgb_pred = xgb.predict(X_test)
print('XGBoost Score:', r2_score(y_test, xgb_pred))
print('XGBoost MSE:', mean_squared_error(y_test, xgb_pred))
