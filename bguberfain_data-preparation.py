import numpy as np

import pandas as pd

import xgboost as xgb
data = pd.read_csv('../input/No-show-Issue-Comma-300k.csv', parse_dates=['ApointmentData', 'AppointmentRegistration'])



# Save target column

y = (data['Status'] == 'No-Show') * 1



# Categorical value to numeric

data['IsMale'] = (data['Gender'] == 'M') * 1



# Drop redundant columns and y

data.drop(['DayOfTheWeek', 'AwaitingTime', 'Gender', 'Status'], axis=1, inplace=True)

data.head()
# Simple feature engenearing (will recreate some of redundand columns)

data['AppointmentRegistration_dow'] = data.AppointmentRegistration.dt.dayofweek

data['AppointmentRegistration_hour'] = data.AppointmentRegistration.dt.hour

data['AppointmentRegistration_month'] = data.AppointmentRegistration.dt.month



data['ApointmentData_dow'] = data.ApointmentData.dt.dayofweek

data['ApointmentData_month'] = data.ApointmentData.dt.month



data['WaitingDays'] = (data.ApointmentData-data.AppointmentRegistration).dt.days



# Now we can drop the date columns

data.drop(['AppointmentRegistration', 'ApointmentData'], axis=1, inplace=True)



data.head()
X = data.values



dtrain = xgb.DMatrix(X, y, feature_names=data.columns)



xgb_params = {

    'objective': 'binary:logistic',

    'eta': 0.1,

    'max_depth': 4,

    'subsample': 0.7,

    'colsample_bytree': 0.7

}



cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=20, nfold=4)
cv_result['test-error-mean'].mean()