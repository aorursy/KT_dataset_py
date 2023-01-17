# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(7)


import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Loading train and test data

df_train = pd.read_csv( '../input/train_V2.csv')
df_train = df_train[df_train['maxPlace'] > 1]

print(df_train.shape)

df_test = pd.read_csv( '../input/test_V2.csv')

print(df_test.shape)
# Removing Id related and categorical columns

target = 'winPlacePerc'
features = list(df_train.columns)
features.remove("Id")
features.remove("matchId")
features.remove("groupId")

features.remove("matchType")

y_train = np.array(df_train[target])
features.remove(target)
x_train = df_train[features]

x_test = df_test[features]

print(x_test.shape,x_train.shape,y_train.shape)
# Split the train and the validation set for the fitting
random_seed=1
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)
m3 = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1)
%%time
m3.fit(x_train, y_train)
print('mae train: ', mean_absolute_error(m3.predict(x_train), y_train))
print('mae val: ', mean_absolute_error(m3.predict(x_val), y_val))
%%time
pred = m3.predict(x_test)
df_test['winPlacePerc'] = pred
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission_rf.csv', index=False)
plt.hist(pred)
plt.hist(y_train)
sub_full_data['PredictedLogRevenue'] = sub_preds
sub_full_data[['PredictedLogRevenue']].to_csv('new_test.csv', index=True)