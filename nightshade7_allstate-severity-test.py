# Code by Abhishek Sehgal

# I've tried to use xgboost for regression

# I mainly work with classification tasks. This is my

# first attempt at a regression task



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from xgboost import XGBRegressor as xgbr

import pylab as pl

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_absolute_error



%pylab inline
df = pd.read_csv("../input/train.csv")

dfSize = df.shape



nFeat = dfSize[1]

cols = list(df.columns.values)

encoder = LabelEncoder()



catData = encoder.fit_transform(df[cols[1]])



for i in range(2,117):

	temp =  encoder.fit_transform(df[cols[i]])

	catData = np.vstack((catData, temp))



data = np.hstack((catData.T, df[cols[117:nFeat-1]].as_matrix()))

output = np.array(df[cols[nFeat-1]])
X_train, X_test, y_train, y_test = train_test_split(data, output, 

                                                    test_size = 0.25, random_state = 42)
gbm = xgbr(base_score=0.5, 

           colsample_bylevel=1, 

           colsample_bytree=0.05, 

           gamma=1,

           learning_rate=0.05,

           max_delta_step=0,

           max_depth=14,

           min_child_weight=13,

           missing=None,

           n_estimators=87,

           nthread=2,

           objective='reg:linear',

           reg_alpha=0,

           reg_lambda=1,

           scale_pos_weight=1,

           seed=0,

           silent=True,

           subsample=0.901345202299914)

gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)
print(r2_score(y_test, y_pred))
gbm.fit(data, output)
df = pd.read_csv("../input/test.csv")

dfSize = df.shape



nFeat = dfSize[1]

cols = list(df.columns.values)

encoder = LabelEncoder()



catData = encoder.fit_transform(df[cols[1]])



for i in range(2,117):

	temp =  encoder.fit_transform(df[cols[i]])

	catData = np.vstack((catData, temp))



data = np.hstack((catData.T, df[cols[117:nFeat]].as_matrix()))
pred = gbm.predict(data)

submission = pd.DataFrame()

submission['id'] = df['id']

submission['loss'] = pred

submission.to_csv('submit1.csv', index=False)