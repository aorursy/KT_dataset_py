# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import gc

import os

import sys



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import RandomForestRegressor



import lightgbm as lgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import dataset

df_train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

df_test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')



# Show some data

df_train.head()

df_train.describe()

# Check row with NaN value

df_train[df_train['winPlacePerc'].isnull()]



# Drop row with NaN 'winPlacePerc' value

df_train.drop(2744604, inplace=True)

df_train['kills'].value_counts()

df_train['DBNOs'].value_counts()

df_train['weaponsAcquired'].value_counts()
# Split the train and the test

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



# Split the train and the validation set for the fitting

random_seed=1

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.05, random_state=random_seed)

# Random Forest

RF = RandomForestRegressor(n_estimators=10, min_samples_leaf=3, max_features=0.5, n_jobs=-1)

RF.fit(x_train, y_train)

print('mae train: ', mean_absolute_error(RF.predict(x_train), y_train))

print('mae val: ', mean_absolute_error(RF.predict(x_val), y_val))

pred = RF.predict(x_test)

df_test['winPlacePerc'] = pred

submission = df_test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)

plt.hist(pred)

plt.hist(y_train)