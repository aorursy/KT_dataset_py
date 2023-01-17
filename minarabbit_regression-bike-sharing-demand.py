import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)



import os

print(os.listdir("../input"))
bike_df = pd.read_csv('../input/train.csv')

bike_df.shape
bike_df.info()
bike_df.head()
# Transform string into datetime type

bike_df['datetime'] = bike_df['datetime'].apply(pd.to_datetime)



# Extract year, month, day and time from the datetime type

bike_df['year'] = bike_df['datetime'].apply(lambda x : x.year)

bike_df['month'] = bike_df['datetime'].apply(lambda x : x.month)

bike_df['day'] = bike_df['datetime'].apply(lambda x : x.day)

bike_df['hour'] = bike_df['datetime'].apply(lambda x : x.hour)

bike_df.head()
bike_df.drop(['datetime', 'casual', 'registered'], axis=1, inplace=True)
from sklearn.metrics import mean_squared_error, mean_absolute_error



# Calculate RMSLE (Root Mean Square Log Error) with using not log() but log1p() due to issues including NaN

def rmsle(y, pred):

    log_y = np.log1p(y)

    log_pred = np.log1p(pred)

    squared_error = (log_y - log_pred) ** 2

    rmsle = np.sqrt(np.mean(squared_error))

    return rmsle



# Calculate RMSE

def rmse(y, pred):

    return np.sqrt(mean_squared_error(y, pred))



# Calculate MSE, RMSE and RMSLE

def evaluate_regr(y, pred):

    rmsle_val = rmsle(y, pred)

    rmse_val = rmse(y, pred)

    mse_val = mean_absolute_error(y, pred)

    print('RMSLE: {:.3f}, RMSE: {:.3f}, MSE: {:.3f}'.format(rmsle_val, rmse_val, mse_val))
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso



y_target = bike_df['count']

X_features = bike_df.drop(['count'], axis=1, inplace=False)



X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=2019)



lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

pred = lr_reg.predict(X_test)



evaluate_regr(y_test, pred)
def get_top_error_data(y_test, pred, n_tops=5):

    result_df = pd.DataFrame(y_test.values, columns=['real_count'])

    result_df['predicted_count'] = np.round(pred)

    result_df['diff'] = np.abs(result_df['real_count'] - result_df['predicted_count'])

    

    print(result_df.sort_values('diff', ascending=False)[:n_tops])

    

get_top_error_data(y_test, pred, n_tops=5)
y_target.hist()
y_log_transform = np.log1p(y_target)

y_log_transform.hist()
y_target_log = np.log1p(y_target)



X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log, test_size=0.3, random_state=2019)



lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

pred = lr_reg.predict(X_test)



# Convert the transformed y_test values into the original values

y_test_exp = np.expm1(y_test)



# Convert the transformed predicted values into the original values

pred_exp = np.expm1(pred)



evaluate_regr(y_test_exp, pred_exp)
coef = pd.Series(lr_reg.coef_, index=X_features.columns)

coef_sort = coef.sort_values(ascending=False)

sns.barplot(x=coef_sort.values, y=coef_sort.index)
X_features_ohe = pd.get_dummies(X_features, columns=['year', 'month', 'hour', 'holiday', 'workingday', 'season', 'weather'])
X_features_ohe.head()
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.3, random_state=2019)



def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    if is_expm1:

        y_test = np.expm1(y_test)

        pred = np.expm1(pred)

    print('###', model.__class__.__name__, '###')

    evaluate_regr(y_test, pred)



lr_reg = LinearRegression()

ridge_reg = Ridge(alpha=10)

lasso_reg = Lasso(alpha=0.01)



for model in [lr_reg, ridge_reg, lasso_reg]:

    get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=True)
coef = pd.Series(lr_reg.coef_, index=X_features_ohe.columns)

coef_sort = coef.sort_values(ascending=False)[:15]

sns.barplot(x=coef_sort.values, y=coef_sort.index)
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.3, random_state=2019)



from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



rf_reg = RandomForestRegressor(n_estimators=500)

gbm_reg = GradientBoostingRegressor(n_estimators=500)

xgb_reg = XGBRegressor(n_estimators=500)

lgbm_reg = LGBMRegressor(n_estimators=500)



for model in [rf_reg, gbm_reg, xgb_reg, lgbm_reg]:

    get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=True)
submission = pd.read_csv('../input/sampleSubmission.csv')
submission.shape
submission.head()
X_test = pd.read_csv('../input/test.csv')

X_test.head()
X_test.shape
# Transform string into datetime type

X_test['datetime'] = X_test['datetime'].apply(pd.to_datetime)



# Extract year, month, day and time from the datetime type

X_test['year'] = X_test['datetime'].apply(lambda x : x.year)

X_test['month'] = X_test['datetime'].apply(lambda x : x.month)

X_test['day'] = X_test['datetime'].apply(lambda x : x.day)

X_test['hour'] = X_test['datetime'].apply(lambda x : x.hour)

X_test.head()
X_test.drop(['datetime'], axis=1, inplace=True)

X_test.head()
X_test.shape
X_test_ohe = pd.get_dummies(X_test, columns=['year', 'month', 'hour', 'holiday', 'workingday', 'season', 'weather'])

X_test_ohe.head()
prediction = lgbm_reg.predict(X_test_ohe)
prediction
submission['count'] = np.round(prediction, 0).astype(int)
submission.head()
submission.to_csv('./My_submission.csv', index=False)