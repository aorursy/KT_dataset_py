# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')
dat_path = '/kaggle/input/house-prices-advanced-regression-techniques/'
train_raw = pd.read_csv(f'{dat_path}train.csv')

test_raw = pd.read_csv(f'{dat_path}test.csv')
sample_sub = pd.read_csv(f'{dat_path}sample_submission.csv')
train_raw.shape
test_raw.shape
train_raw.dropna(axis=1)
test_raw.dropna(axis=1)
plt.rcParams['figure.figsize'] = [12, 8]

plt.rcParams['font.size'] = 16
plt.bar(np.arange(test_raw.shape[1]), test_raw.isna().sum() / len(test_raw))

plt.ylabel('Fraction of Rows with NaN Value')

plt.xlabel('Column Index')
plt.bar(np.arange(train_raw.shape[1]), train_raw.isna().sum() / len(train_raw))

plt.ylabel('Fraction of Rows with NaN Value')

plt.xlabel('Column Index')
test_raw.shape
test = test_raw.dropna(thresh=test_raw.shape[0]*0.9, axis=1)

train = train_raw.dropna(thresh=train_raw.shape[0]*0.9, axis=1)
test.shape
train.shape
train = train.dropna(axis=0)
train.shape
train.columns.values
train['Exterior2nd'].dtype.name
categorical_train_cols = [col_name for col_name in train.columns.values if train[col_name].dtype.name == 'object']
categorical_train_cols += ['MSSubClass']
categorical_test_cols = [col_name for col_name in test.columns.values if test[col_name].dtype.name == 'object']
categorical_test_cols += ['MSSubClass']
numeric_cols = [col_name for col_name in train.columns.values if col_name not in categorical_train_cols]
train_df = pd.concat((train[numeric_cols], pd.concat([

    pd.get_dummies(train[col_name], prefix = f'{col_name}') for col_name in categorical_train_cols

], axis = 1)), axis = 1)
numeric_test_cols = [col_name for col_name in test.columns.values if col_name not in categorical_test_cols]
test_df = pd.concat((test[numeric_test_cols], pd.concat([

    pd.get_dummies(test[col_name], prefix = f'{col_name}') for col_name in categorical_test_cols

], axis = 1)), axis = 1)
train_df
test_df
extra_train_cols = set(train_df.columns.values).difference(set(test_df.columns.values))
extra_test_cols = set(test_df.columns.values).difference(set(train_df.columns.values))
extra_train_cols.remove('SalePrice')
train_df = train_df.drop(columns = extra_train_cols)

test_df = test_df.drop(columns = extra_test_cols)
train_X = train_df.copy().drop(columns = ['SalePrice', 'Id'])

train_Y = train_df['SalePrice'].copy()
from sklearn.model_selection import train_test_split
train_X, test_X_all, train_Y, test_Y_all = train_test_split(train_X, train_Y, train_size=0.7, shuffle=True, random_state = 42)
train_X.shape
test_X_all.shape
submission_test_df = test_df.copy()
test_X, validation_X, test_Y, validation_Y = train_test_split(test_X_all, test_Y_all, train_size=0.6, shuffle=True)
test_X.shape
validation_X.shape
plt.hist(train_Y, bins = 40)

plt.title('Distribution of Sale Prices for Train Data')
train_log_Y = np.log(train_Y)
plt.hist(train_log_Y, bins = 40)

plt.title('Distribution of (Log-Scaled) Sale Prices for Train Data')
correlations_series = train_df.corrwith(train_log_Y, method='pearson').dropna()
correlations_series
sorted(correlations_series)
plt.bar(np.arange(len(correlations_series)), sorted(correlations_series))

plt.title('Correlation of Individual Features with Target Variable (LogSalePrice)')

plt.ylabel('Correlation (Pearson R)')

plt.xlabel('Feature Index Number');
from numpy.linalg import lstsq, norm
# set rcond = -1 to use higher precision than the default

lstsq_weights, residuals, train_rank, train_sing_values = lstsq(train_X, train_log_Y, rcond= -1)
lstsq_train_loss = norm(train_X.dot(lstsq_weights) - train_log_Y)**2 / len(train_X)
lstsq_train_loss
norm(np.exp(train_X.dot(lstsq_weights)) - train_Y) / len(train_X)
test_log_Y = np.log(test_Y)
lstsq_test_loss = norm(test_X.dot(lstsq_weights) - test_log_Y) / len(test_log_Y) 
lstsq_test_loss
norm(np.exp(test_X.dot(lstsq_weights)) - test_Y) / len(test_log_Y) 
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 100, max_depth = 10, random_state = 42)
rf_regressor.fit(train_X, train_log_Y)
rf_train_loss = norm(rf_regressor.predict(train_X) - train_log_Y) / len(train_X)
rf_test_loss = norm(rf_regressor.predict(test_X) - test_log_Y) / len(test_Y)
rf_train_loss
rf_test_loss
weak_rf_regressor = RandomForestRegressor(n_estimators = 10, max_depth = 4, random_state = 42)
weak_rf_regressor.fit(train_X, train_log_Y)
weak_rf_train_loss = norm(weak_rf_regressor.predict(train_X) - train_log_Y) / len(train_X)
weak_rf_test_loss = norm(weak_rf_regressor.predict(test_X) - test_log_Y) / len(test_X)
weak_rf_train_loss
weak_rf_test_loss
plt.bar(np.arange(len(lstsq_weights)), sorted(np.abs(lstsq_weights)))

plt.title('Feature Weights for Ordinary Least Squares Regression')

plt.ylabel('Coefficient (Absolute Value)');
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha = 1.0, normalize = True, fit_intercept = True, tol=1e-6, random_state = 42)
lasso_model.fit(train_X, train_log_Y)
lasso_train_loss = norm(lasso_model.predict(train_X) - train_log_Y) / len(train_X)

lasso_test_loss = norm(lasso_model.predict(test_X) - test_log_Y) / len(test_X)
lasso_train_loss
lasso_test_loss
lasso_model_001 = Lasso(alpha = 0.01, normalize = True, fit_intercept = True, tol=1e-6, random_state = 42)
lasso_model_001.fit(train_X, train_log_Y)
lasso_train_loss_001 = norm(lasso_model_001.predict(train_X) - train_log_Y) / len(train_X)

lasso_test_loss_001 = norm(lasso_model_001.predict(test_X) - test_log_Y) / len(test_X)
lasso_train_loss_001
lasso_test_loss_001
lasso_model_1e4 = Lasso(alpha = 0.0001, normalize = True, fit_intercept = True, tol=1e-6, random_state = 42)
lasso_model_1e4.fit(train_X, train_log_Y)
lasso_train_loss_1e4 = norm(lasso_model_1e4.predict(train_X) - train_log_Y) / len(train_X)

lasso_test_loss_1e4 = norm(lasso_model_1e4.predict(test_X) - test_log_Y) / len(test_X)
lasso_train_loss_1e4
lasso_test_loss_1e4
lstsq_test_loss
lasso_weights = lasso_model_1e4.coef_
plt.scatter(np.abs(lstsq_weights), np.abs(lasso_weights), s = 10, marker='o');

plt.xlabel('Feature Weight for Least Squares')

plt.ylabel('Feature Weight for LASSO')

plt.title('Feature Weights for Regression with/without Regularization');
validation_log_Y = np.log(validation_Y)
lasso_validation_loss_1e4 = norm(lasso_model_1e4.predict(validation_X) - validation_log_Y) / len(validation_X)

weak_rf_regressor_validation_loss = norm(weak_rf_regressor.predict(validation_X) - validation_log_Y) / len(validation_X)

rf_regressor_validation_loss = norm(rf_regressor.predict(validation_X) - validation_log_Y) / len(validation_X)
lstsq_validation_loss = norm(validation_X.dot(lstsq_weights) - validation_log_Y) / len(validation_X)
lasso_validation_loss_1e4
weak_rf_regressor_validation_loss
rf_regressor_validation_loss
lstsq_validation_loss
plt.bar(np.arange(len(rf_regressor.feature_importances_)), rf_regressor.feature_importances_)

plt.ylabel('Feature Importance (Sums to 1)')

plt.title('Feature Importance for Random Forest Regressor')
rf_weights = rf_regressor.feature_importances_
plt.plot(np.arange(len(rf_weights)), np.cumsum(sorted(rf_weights, reverse=True)), marker='^')

plt.title('Cumulative Feature Weight for Random Forest')

plt.ylabel('Sum of Feature Weights')

plt.xlabel('Index of feature weight')
top_feature_indices = np.where(rf_weights > 0.01)
train_X.columns.values[top_feature_indices]
rf_weights[top_feature_indices]
plt.bar(np.arange(len(top_feature_indices[0])), rf_weights[top_feature_indices])

plt.xticks(np.arange(len(top_feature_indices[0])), train_X.columns.values[top_feature_indices], rotation=60);

plt.ylabel('Feature Weight')

plt.title('Weights for Top Features in Random Forest Regressor');
sample_submission_df = pd.read_csv(f'{dat_path}sample_submission.csv')
submission_test_df.shape
train_X.shape
submission_X = submission_test_df.drop(columns = ['Id'])
feature_means = np.mean(train_X, axis=0)
submission_X_no_nan = submission_X.fillna(value=feature_means)
submission_X_no_nan.shape
submission_Y_predict = np.exp(rf_regressor.predict(submission_X_no_nan))
submission_df_final = pd.concat((submission_test_df['Id'], pd.Series(submission_Y_predict)), axis = 1)
submission_df_final.rename(columns = {0: 'SalePrice'}, inplace=True)
submission_df_final
submission_df_final.to_csv('house_prices_submission.csv')