# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.linear_model import LinearRegression as LR

from sklearn.neural_network import MLPRegressor as MLPR
data_dir = '../input/ieee-pes-bdc-datathon-year-2020'

df = pd.read_csv(f'{data_dir}/train.csv')

test_df = pd.read_csv(f'{data_dir}/test.csv')
df.head(5)
data_len = len(df)

pct = 1.0 # change it to 0.8~0.9

train_len = int(1.0*data_len)

train_df = df[:train_len]

val_df = df[train_len:]
X_train = train_df.drop(['ID', 'global_horizontal_irradiance'], axis=1).values.reshape(-1, 6)

y_train = train_df['global_horizontal_irradiance'].values.reshape(len(train_df))
X_val = val_df.drop(['ID', 'global_horizontal_irradiance'], axis=1).values.reshape(-1, 6)

y_val = val_df['global_horizontal_irradiance'].values.reshape(len(val_df))
X_test = test_df.drop(['ID'], axis=1).values.reshape(-1, 6)

test_ID = test_df['ID'].values.reshape(len(test_df))
reg = LR(normalize=True)

reg.fit(X_train, y_train)
reg.coef_
preds = reg.predict(X_test)
regr = MLPR(random_state=1, hidden_layer_sizes = (32, 8, 2), max_iter=5, validation_fraction=0.1, learning_rate_init=0.02, verbose=True)

regr.fit(X_train, y_train)
preds = regr.predict(X_test)

preds = [0 if p<0 else p for p in preds] # Since GHI can not be less than 0
zippedList =  list(zip(test_ID, preds))

submission = pd.DataFrame(zippedList, columns = ['ID','global_horizontal_irradiance'])

submission.to_csv('submission.csv', index=False)
submission.head(5)