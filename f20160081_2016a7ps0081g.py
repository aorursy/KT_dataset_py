# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from xgboost import plot_importance

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel

from numpy import sort

import seaborn as sns

from sklearn.externals import joblib



data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



data.head()
X = data.drop(['AveragePrice'], axis=1)



y = data['AveragePrice']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)



# model = XGBRegressor(n_jobs=-1, n_estimators=100, max_depth=15)

# model.fit(X_train, 

#             y_train,

#             verbose=True)
#FIt, predict and save to file

model = XGBRegressor(n_jobs=4, learning_rate=0.05, max_depth=8, min_child_weight=0.5, n_estimators=1300)

model.fit(X, 

            y,

            verbose=True)

test_predict_optimal = model.predict(test)

new_df = pd.DataFrame({"id": test['id'] , "AveragePrice":test_predict_optimal})

new_df.to_csv("XGB_manual_custom_5.csv", index=False)
predictions = model.predict(X_val)

scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)

print(scores)

print('Mean Absolute Error: %2f' %(-1 * scores.mean()))
mae = mean_absolute_error(predictions, y_val)

print("Mean Absolute Error : " + str(mae))



error_percent = mae/data['AveragePrice'].mean()*100

print(str(error_percent) + ' %')



#Done manually

#11.205524360061048 for 600

#11.262510333172916 for 500

#11.068815661585504 for 800

 #10.983674139407537  for 1000

#10.963561889818891 for 1200

#10.957846846078342 for 1300 - max depth = 3

#10.973262601342018 for 1500



##10.165696679252953 for 1300 - max depth = 5

##9.88223441450899 for 1300 - max depth = 6

##9.728417154369875 for 1300 - max depth = 7

#9.450882068676028 % for 1300 max dpth  10

#9.335617887568638 % for 1300 max dpth  15

#9.3499321261874638 % for 1300 max dpth  20

#