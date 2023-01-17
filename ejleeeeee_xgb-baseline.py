# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/ds2-ds5-competition-2/train.csv")

test_input = pd.read_csv("../input/ds2-ds5-competition-2/test_input.csv")

test = pd.read_csv("../input/ds2-ds5-competition-2/test.csv")

submission = pd.read_csv("../input/ds2-ds5-competition-2/sample_submission.csv")
train.head()
train['month'] = train['timestamp'].str[5:7]

train['time'] = train['timestamp'].str[11:13]

train['timestamp'] = pd.to_datetime(train['timestamp'])
mean_usage_by_id = train[['device_id', 'value']].groupby("device_id", as_index=False).mean()

mean_usage_by_id.describe()
ax = sns.lineplot(x="month", y="value", data=train)
sns.set()

plt.figure(figsize=(10, 7))

ax = sns.lineplot(x="time", y="value", hue="month", data=train, palette = sns.color_palette("hls", 12))
p_start = '2013-09-01 00:00:00'

p_end = '2013-10-01 00:00:00'

new_train = train.set_index('timestamp')[p_start:p_end]

plt.figure(figsize=(10, 7))

ax = sns.lineplot(x="time", y="value", data=new_train)
set(test_input['timestamp'])
def window_stack(a, stepsize=24, width=144):

    n = a.shape[0]

    nd = np.empty((0, width))

    for i in range(0,(int(n/stepsize) - int(width/stepsize - 1))):

        nd = np.vstack([nd, a[(i*stepsize):(i*stepsize + width)]])

    return nd
train = train.sort_values(by=['device_id', 'timestamp'], axis=0).reset_index(drop=True)



train_usage = np.empty((0, 144))

id_set = list(set(train['device_id']))

for id_ in id_set:

    one_id_usage = train.loc[train['device_id'] == id_]['value'].reset_index(drop=True)

    train_usage = np.concatenate((train_usage, window_stack(one_id_usage)), axis=0)
test_input_usage = np.empty((0, 120))

id_set = list(set(test_input['device_id']))

for id_ in id_set:

    one_id_usage = test_input.loc[test_input['device_id'] == id_]['value'].reset_index(drop=True)

    test_input_usage = np.concatenate((test_input_usage, window_stack(one_id_usage, stepsize=120, width=120)), axis=0)
id_order = np.concatenate([[i]*96 for i in id_set])
train_usage.shape
test_input_usage.shape
import xgboost as xgb

from sklearn.metrics import mean_squared_error

from sklearn.multioutput import MultiOutputRegressor



X_train, y_train = np.split(train_usage, (120, ), axis=1)

X_test = test_input_usage



multioutputregressor_trvalte = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10))

multioutputregressor_trvalte.fit(X_train, y_train)

predictions = multioutputregressor_trvalte.predict(X_test)



y_pred = np.concatenate(predictions)
pred_df = pd.DataFrame({'device_id' : id_order,

                        'timestamp' : test['timestamp'],

                       'value' : y_pred})

pred_df = pd.merge(test, pred_df, on=['device_id', 'timestamp'])
submission['value'] = pred_df['value'].values

submission.to_csv('submission_xgb.csv', index=False)
avg_usage = np.mean(X_train, axis=1)



X_train = np.divide(X_train, avg_usage[:,None])

y_train = np.divide(y_train, avg_usage[:,None])



avg_usage_test = np.mean(X_test, axis=1)

X_test = np.divide(X_test, avg_usage_test[:,None])
multioutputregressor_trvalte = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10))

multioutputregressor_trvalte.fit(X_train, y_train)

nor_predictions = multioutputregressor_trvalte.predict(X_test)



predictions = np.multiply(nor_predictions, avg_usage_test[:,None])

y_pred = np.concatenate(predictions)
pred_df = pd.DataFrame({'device_id' : id_order,

                        'timestamp' : test['timestamp'],

                       'value' : y_pred})

pred_df = pd.merge(test, pred_df, on=['device_id', 'timestamp'])
submission['value'] = pred_df['value'].values

submission.to_csv('submission_xgb_nor.csv', index=False)