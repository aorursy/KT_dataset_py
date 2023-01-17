import numpy as np

import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/ml-academy-competition-2020/train.csv')

df.head()
df_test = pd.read_csv('/kaggle/input/ml-academy-competition-2020/test.csv',

                      index_col='index')

df_test.head()
df_sample_submission = pd.read_csv('/kaggle/input/ml-academy-competition-2020/sample_submission.csv',

                                   index_col='index')

df_sample_submission
X = df.drop('price', axis=1)

y = df['price']



X_test = df_test
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_absolute_error
model = DummyRegressor(strategy="mean")

model.fit(X_train, y_train)



y_train_pred = model.predict(X_train)

y_val_pred = model.predict(X_val)



train_score = mean_absolute_error(y_train, y_train_pred)

val_score = mean_absolute_error(y_val, y_val_pred)



print("Baseline\ntrain MAE: {:.10}\nvalidation MAE: {:.10}".format(train_score, val_score))
model.fit(X, y)
y_test_pred = model.predict(X_test)

y_test_pred
y_test_pred_rounded = (y_test_pred / 100).round(0).astype(int) * 100

y_test_pred_rounded
df_submission = pd.DataFrame(y_test_pred_rounded, columns=['price'])

df_submission.index.name = 'index'

df_submission
df_submission.to_csv('/kaggle/working/my_submission.csv')