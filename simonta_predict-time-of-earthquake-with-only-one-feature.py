import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

%matplotlib inline
df_train_10M = pd.read_csv('../input/train.csv', nrows=200000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

df_train_10M.head()
df_train_10M["index_obs"] = (df_train_10M.index.astype(float)/150000).astype(int)
train_set = df_train_10M.groupby('index_obs').agg({'acoustic_data': 'std', 'time_to_failure': 'mean'})
train_set.columns = ['acoustic_data_std', 'time_to_failure_mean']

train_set.head()
del df_train_10M
train_set[train_set['time_to_failure_mean'].diff() > 0].head()
train_set["acoustic_data_transform"] = train_set["acoustic_data_std"].clip(-20, 12).rolling(10, min_periods=1).median()
fig, ax1 = plt.subplots(figsize=(20,12))

ax2 = ax1.twinx()

ax1.plot(train_set["acoustic_data_transform"])

plt.axvline(x=38, color='r')

plt.axvline(x=334, color='r')

plt.axvline(x=697, color='r')

plt.title('Smoothed standard deviation of acoustic_data vs. time', size=20)

plt.show()
from sklearn.linear_model import LinearRegression

regr = LinearRegression()

regr.fit(train_set[["acoustic_data_transform"]], train_set["time_to_failure_mean"])



print('Coefficients: \n', regr.coef_)

print('Intercept: \n', regr.intercept_)
submission_file = pd.read_csv('../input/sample_submission.csv')

submission_file.head()
for index, seg_id in enumerate(submission_file['seg_id']):

    seg = pd.read_csv('../input/test/' + str(seg_id) + '.csv')

    x = seg['acoustic_data'].values

    std_x = max(-20, min(12, np.std(x)))

    submission_file.loc[index, "time_to_failure"] = max(0, regr.intercept_ + regr.coef_ * std_x)

    del seg
submission_file.to_csv('submission.csv', index=False)