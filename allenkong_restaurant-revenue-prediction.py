%matplotlib inline

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

from scipy import stats

from scipy.stats import norm, skew

from sklearn import preprocessing

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor, plot_importance

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')

df.shape
test_df = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')

test_df.shape
df.head()
df.describe()
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)



display_all(df.head().transpose())
df.isnull().sum().sort_index()/len(df)
fig, ax = plt.subplots(1,2, figsize=(19, 5))

g1 = sns.countplot(df['Type'],palette="Set2", ax=ax[0]);

g2 = sns.countplot(test_df['Type'],palette="Set2", ax=ax[1]);

fig.show()
fig, ax = plt.subplots(1,2, figsize=(19, 5))

g1 = sns.countplot(df['City Group'],palette="Set2", ax=ax[0]);

g2 = sns.countplot(test_df['City Group'],palette="Set2", ax=ax[1]);

fig.show()
df['City'].nunique()
test_df['City'].nunique()
test_df.loc[test_df['Type']=='MB', 'Type'] = 'DT'
df.drop('City', axis=1, inplace=True)

test_df.drop('City', axis=1, inplace=True)
import datetime

df.drop('Id',axis=1,inplace=True)

df['Open Date']  = pd.to_datetime(df['Open Date'])

test_df['Open Date']  = pd.to_datetime(test_df['Open Date'])

launch_date = datetime.datetime(2015, 3, 23)

df['Days Open'] = (launch_date - df['Open Date']).dt.days

test_df['Days Open'] = (launch_date - test_df['Open Date']).dt.days

df.drop('Open Date', axis=1, inplace=True)

test_df.drop('Open Date', axis=1, inplace=True)
df.dtypes
(mu, sigma) = norm.fit(df['revenue'])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 5))

ax1 = sns.distplot(df['revenue'] , fit=norm, ax=ax1)

ax1.legend([f'Normal distribution ($\mu=$ {mu:.3f} and $\sigma=$ {sigma:.3f})'], loc='best')

ax1.set_ylabel('Frequency')

ax1.set_title('Revenue Distribution')

ax2 = stats.probplot(df['revenue'], plot=plt)

f.show();
# Revenue is right skewed, taking the log will make it more normally distributed for the linear models

# Remember to use expm1 on predictions to transform back to dollar amount

(mu, sigma) = norm.fit(np.log1p(df['revenue']))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 5))

ax1 = sns.distplot(np.log1p(df['revenue']) , fit=norm, ax=ax1)

ax1.legend([f'Normal distribution ($\mu=$ {mu:.3f} and $\sigma=$ {sigma:.3f})'], loc='best')

ax1.set_ylabel('Frequency')

ax1.set_title('Log(1+Revenue) Distribution')

ax2 = stats.probplot(np.log(df['revenue']), plot=plt)

f.show();
# Correlation between numeric features with revenue

plt.figure(figsize=(10, 8))

sns.heatmap(df.drop(['revenue','City Group','Type'], axis=1).corr(), square=True)

plt.suptitle('Pearson Correlation Heatmap')

plt.show();
corr_with_revenue = df.drop(['City Group','Type'],axis=1).corr()['revenue'].sort_values(ascending=False)

plt.figure(figsize=(10,7))

corr_with_revenue.drop('revenue').plot.bar()

plt.show();
sns.pairplot(df[df.corr()['revenue'].sort_values(ascending=False).index[:5]])

plt.show();
numeric_features = df.dtypes[df.dtypes != "object"].index

skewed_features = df[numeric_features].apply(lambda x: skew(x))

skewed_features = skewed_features[skewed_features > 0.5].index

df[skewed_features] = np.log1p(df[skewed_features])

test_df[skewed_features.drop('revenue')] = np.log1p(test_df[skewed_features.drop('revenue')])
# Dummy Encoding for object types, avoiding redundancy of OHE

columnsToEncode = df.select_dtypes(include=[object]).columns

df = pd.get_dummies(df, columns=columnsToEncode, drop_first=True)

test_df = pd.get_dummies(test_df, columns=columnsToEncode, drop_first=True)
X, y = df.drop('revenue', axis=1), df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=118)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso
params_ridge = {

    'alpha' : [.01, .1, .5, .7, .9, .95, .99, 1, 5, 10, 20],

    'fit_intercept' : [True, False],

    'normalize' : [True,False],

    'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

}



ridge_model = Ridge()

ridge_regressor = GridSearchCV(ridge_model, params_ridge, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

ridge_regressor.fit(X_train, y_train)

print(f'Optimal alpha: {ridge_regressor.best_params_["alpha"]:.2f}')

print(f'Optimal fit_intercept: {ridge_regressor.best_params_["fit_intercept"]}')

print(f'Optimal normalize: {ridge_regressor.best_params_["normalize"]}')

print(f'Optimal solver: {ridge_regressor.best_params_["solver"]}')

print(f'Best score: {ridge_regressor.best_score_}')
ridge_model = Ridge(alpha=ridge_regressor.best_params_["alpha"], fit_intercept=ridge_regressor.best_params_["fit_intercept"], 

                    normalize=ridge_regressor.best_params_["normalize"], solver=ridge_regressor.best_params_["solver"])

ridge_model.fit(X_train, y_train)

y_train_pred = ridge_model.predict(X_train)

y_pred = ridge_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
# Ridge Model Feature Importance

ridge_feature_coef = pd.Series(index = X_train.columns, data = np.abs(ridge_model.coef_))

ridge_feature_coef.sort_values().plot(kind = 'bar', figsize = (13,5));
params_lasso = {

    'alpha' : [.01, .1, .5, .7, .9, .95, .99, 1, 5, 10, 20],

    'fit_intercept' : [True, False],

    'normalize' : [True,False],

}



lasso_model = Lasso()

lasso_regressor = GridSearchCV(lasso_model, params_lasso, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

lasso_regressor.fit(X_train, y_train)

print(f'Optimal alpha: {lasso_regressor.best_params_["alpha"]:.2f}')

print(f'Optimal fit_intercept: {lasso_regressor.best_params_["fit_intercept"]}')

print(f'Optimal normalize: {lasso_regressor.best_params_["normalize"]}')

print(f'Best score: {lasso_regressor.best_score_}')
lasso_model = Lasso(alpha=lasso_regressor.best_params_["alpha"], fit_intercept=lasso_regressor.best_params_["fit_intercept"], 

                    normalize=lasso_regressor.best_params_["normalize"])

lasso_model.fit(X_train, y_train)

y_train_pred = lasso_model.predict(X_train)

y_pred = lasso_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
# Lasso Model Feature Importance

lasso_feature_coef = pd.Series(index = X_train.columns, data = np.abs(lasso_model.coef_))

lasso_feature_coef.sort_values().plot(kind = 'bar', figsize = (13,5));
from sklearn.linear_model import ElasticNetCV, ElasticNet



# Use ElasticNetCV to tune alpha automatically instead of redundantly using ElasticNet and GridSearchCV

el_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=1e-2, cv=5, n_jobs=-1)         

el_model.fit(X_train, y_train)

print(f'Optimal alpha: {el_model.alpha_:.6f}')

print(f'Optimal l1_ratio: {el_model.l1_ratio_:.3f}')

print(f'Number of iterations {el_model.n_iter_}')
y_train_pred = el_model.predict(X_train)

y_pred = el_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
# ElasticNet Model Feature Importance

el_feature_coef = pd.Series(index = X_train.columns, data = np.abs(el_model.coef_))

n_features = (el_feature_coef>0).sum()

print(f'{n_features} features with reduction of {(1-n_features/len(el_feature_coef))*100:2.2f}%')

el_feature_coef.sort_values().plot(kind = 'bar', figsize = (13,5));
from sklearn.neighbors import KNeighborsRegressor



params_knn = {

    'n_neighbors' : [3, 5, 7, 9, 11],

}



knn_model = KNeighborsRegressor()

knn_regressor = GridSearchCV(knn_model, params_knn, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

knn_regressor.fit(X_train, y_train)

print(f'Optimal neighbors: {knn_regressor.best_params_["n_neighbors"]}')

print(f'Best score: {knn_regressor.best_score_}')
knn_model = KNeighborsRegressor(n_neighbors=knn_regressor.best_params_["n_neighbors"])

knn_model.fit(X_train, y_train)

y_train_pred = knn_model.predict(X_train)

y_pred = knn_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
from sklearn.ensemble import RandomForestRegressor



params_rf = {

    'max_depth': [5, 10, 30, 100],

    'max_features': [.3, .4, .5, .6],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [30, 100, 500]

}



rf = RandomForestRegressor()

rf_regressor = GridSearchCV(rf, params_rf, scoring='neg_root_mean_squared_error', cv = 5, n_jobs = -1)

rf_regressor.fit(X_train, y_train)

print(f'Optimal depth: {rf_regressor.best_params_["max_depth"]}')

print(f'Optimal max_features: {rf_regressor.best_params_["max_features"]}')

print(f'Optimal min_sample_leaf: {rf_regressor.best_params_["min_samples_leaf"]}')

print(f'Optimal min_samples_split: {rf_regressor.best_params_["min_samples_split"]}')

print(f'Optimal n_estimators: {rf_regressor.best_params_["n_estimators"]}')

print(f'Best score: {rf_regressor.best_score_}')
rf_model = RandomForestRegressor(max_depth=rf_regressor.best_params_["max_depth"], 

                                 max_features=rf_regressor.best_params_["max_features"], 

                                 min_samples_leaf=rf_regressor.best_params_["min_samples_leaf"], 

                                 min_samples_split=rf_regressor.best_params_["min_samples_split"], 

                                 n_estimators=rf_regressor.best_params_["n_estimators"], 

                                 n_jobs=-1, oob_score=True)

rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)

y_pred = rf_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
# Random Forest Model Feature Importance

rf_feature_importance = pd.Series(index = X_train.columns, data = np.abs(rf_model.feature_importances_))

n_features = (rf_feature_importance>0).sum()

print(f'{n_features} features with reduction of {(1-n_features/len(rf_feature_importance))*100:2.2f}%')

rf_feature_importance.sort_values().plot(kind = 'bar', figsize = (13,5));
import lightgbm as lgbm



params_lgbm = {

    'learning_rate': [.01, .1, .5, .7, .9, .95, .99, 1],

    'boosting': ['gbdt'],

    'metric': ['l1'],

    'feature_fraction': [.3, .4, .5, 1],

    'num_leaves': [20],

    'min_data': [10],

    'max_depth': [10],

    'n_estimators': [10, 30, 50, 100]

}



lgb = lgbm.LGBMRegressor()

lgb_regressor = GridSearchCV(lgb, params_lgbm, scoring='neg_root_mean_squared_error', cv = 5, n_jobs = -1)

lgb_regressor.fit(X_train, y_train)

print(f'Optimal lr: {lgb_regressor.best_params_["learning_rate"]}')

print(f'Optimal feature_fraction: {lgb_regressor.best_params_["feature_fraction"]}')

print(f'Optimal n_estimators: {lgb_regressor.best_params_["n_estimators"]}')

print(f'Best score: {lgb_regressor.best_score_}')
lgb_model = lgbm.LGBMRegressor(learning_rate=lgb_regressor.best_params_["learning_rate"], boosting='gbdt', 

                               metric='l1', feature_fraction=lgb_regressor.best_params_["feature_fraction"], 

                               num_leaves=20, min_data=10, max_depth=10, 

                               num_iterations=lgb_regressor.best_params_["n_estimators"], n_jobs=-1)

lgb_model.fit(X_train, y_train)

y_train_pred = lgb_model.predict(X_train)

y_pred = lgb_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
# LightGBM Feature Importance

lgb_feature_importance = pd.Series(index = X_train.columns, data = np.abs(lgb_model.feature_importances_))

n_features = (lgb_feature_importance>0).sum()

print(f'{n_features} features with reduction of {(1-n_features/len(lgb_feature_importance))*100:2.2f}%')

lgb_feature_importance.sort_values().plot(kind = 'bar', figsize = (13,5));
params_xgb = {

    'learning_rate': [.1, .5, .7, .9, .95, .99, 1],

    'colsample_bytree': [.3, .4, .5, .6],

    'max_depth': [4],

    'alpha': [3],

    'subsample': [.5],

    'n_estimators': [30, 70, 100, 200]

}



xgb_model = XGBRegressor()

xgb_regressor = GridSearchCV(xgb_model, params_xgb, scoring='neg_root_mean_squared_error', cv = 5, n_jobs = -1)

xgb_regressor.fit(X_train, y_train)

print(f'Optimal lr: {xgb_regressor.best_params_["learning_rate"]}')

print(f'Optimal colsample_bytree: {xgb_regressor.best_params_["colsample_bytree"]}')

print(f'Optimal n_estimators: {xgb_regressor.best_params_["n_estimators"]}')

print(f'Best score: {xgb_regressor.best_score_}')
xgb_model = XGBRegressor(learning_rate=xgb_regressor.best_params_["learning_rate"], 

                         colsample_bytree=xgb_regressor.best_params_["colsample_bytree"], 

                         max_depth=4, alpha=3, subsample=.5, 

                         n_estimators=xgb_regressor.best_params_["n_estimators"], n_jobs=-1)

xgb_model.fit(X_train, y_train)

y_train_pred = xgb_model.predict(X_train)

y_pred = xgb_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
# XGB with early stopping

xgb_model.fit(X_train, y_train, early_stopping_rounds=4,

             eval_set=[(X_test, y_test)], verbose=False)

y_train_pred = xgb_model.predict(X_train)

y_pred = xgb_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Train RMSE: {train_rmse:.4f}')

print(f'Test RMSE: {test_rmse:.4f}')
# XGB Feature Importance, relevant features can be selected based on its score

feature_important = xgb_model.get_booster().get_fscore()

keys = list(feature_important.keys())

values = list(feature_important.values())



data = pd.DataFrame(data=values, index=keys, columns=['score']).sort_values(by = 'score', ascending=True)

data.plot(kind='bar', figsize = (13,5))

plt.show()
submission = pd.DataFrame(columns=['Id','Prediction'])

submission['Id'] = test_df['Id']



ridge_pred = ridge_model.predict(test_df.drop('Id', axis=1))

submission['Prediction'] = np.expm1(ridge_pred)

submission.to_csv('submission_ridge.csv',index=False)



lasso_pred = lasso_model.predict(test_df.drop('Id', axis=1))

submission['Prediction'] = np.expm1(lasso_pred)

submission.to_csv('submission_lasso.csv',index=False)



elastic_pred = el_model.predict(test_df.drop('Id', axis=1))

submission['Prediction'] = np.expm1(elastic_pred)

submission.to_csv('submission_elastic.csv',index=False)



knn_pred = knn_model.predict(test_df.drop('Id', axis=1))

submission['Prediction'] = np.expm1(knn_pred)

submission.to_csv('submission_knn.csv',index=False)



rf_pred = rf_model.predict(test_df.drop('Id', axis=1))

submission['Prediction'] = np.expm1(rf_pred)

submission.to_csv('submission_rf.csv',index=False)



lgb_pred = lgb_model.predict(test_df.drop('Id', axis=1))

submission['Prediction'] = np.expm1(lgb_pred)

submission.to_csv('submission_lgb.csv',index=False)



xgb_pred = xgb_model.predict(test_df.drop('Id', axis=1))

submission['Prediction'] = np.expm1(xgb_pred)

submission.to_csv('submission_xgb.csv',index=False)