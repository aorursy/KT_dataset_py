import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost

print(check_output(["ls", "../input"]).decode("utf8"))
DATA_HOME_DIR = '../input'
air_visit = pd.read_csv(os.path.join(DATA_HOME_DIR, 'air_visit_data.csv'))

date_info = pd.read_csv(os.path.join(DATA_HOME_DIR, 'date_info.csv'))
air_visit_date = pd.merge(air_visit, date_info, how='left', left_on='visit_date', right_on='calendar_date')

one_hot = pd.get_dummies(air_visit_date['day_of_week'])

X_train_all = air_visit_date[['holiday_flg']].join(one_hot)

y_train_all = air_visit_date['visitors']
validation = 0.1

mask = np.random.rand(len(X_train_all)) < validation

X_train = X_train_all[~mask]

y_train = y_train_all[~mask]

X_validation = X_train_all[mask]

y_validation = y_train_all[mask]
# we will simply use 

xgb = xgboost.XGBRegressor()
xgb.fit(X_train, y_train)
y_test = xgb.predict(X_validation)
rmsle = np.sqrt(np.average(np.log(y_test + 1)**2 - np.log(y_validation + 1)**2))

print(rmsle)
plt.scatter(y_validation, y_test)

plt.xlabel("Visitor (actual)")

plt.ylabel("Visitor (predicted)")

plt.show()
xgb = xgboost.XGBRegressor()

xgb.fit(X_train_all, y_train_all)
submission = pd.read_csv(os.path.join(DATA_HOME_DIR, 'sample_submission.csv'))

air_store_id = ['_'.join(id.split('_')[:2]) for id in submission['id']]

visit_date = [id.split('_')[2] for id in submission['id']]

air_visit_test = pd.DataFrame({'air_store_id': air_store_id, 'visit_date': visit_date})

air_visit_date_test = pd.merge(air_visit_test, date_info, how='left', left_on='visit_date', right_on='calendar_date')

one_hot = pd.get_dummies(air_visit_date_test['day_of_week'])

X_test = air_visit_date_test[['holiday_flg']].join(one_hot)
y_test = xgb.predict(X_test)

submission.visitors = y_test
submission.to_csv('submission.csv', index=False)