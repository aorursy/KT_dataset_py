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



df_submission=pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')



df_test=pd.read_csv('../input/tmdb-box-office-prediction/test.csv')

df_submission.head(1), df_test.head(1)

#insert data with 78 features and target

df_tmdb = pd.read_csv("../input/tmdb-2/tmdb.csv")

df_tmdb.head(5)
#insert some libraries

import json

import ast

from pprint import pprint

import seaborn as sns 

from scipy.stats import norm,skew

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import numpy as np

pd.set_option('display.max_columns', None)
#split tmdb into train and test 

df_train= df_tmdb.iloc[0:3000] # first 3000 rows of the tmdb dataframe

df_test=df_tmdb.iloc[3000:7398]
df_train.shape, df_test.shape
#split data into features and target. All features are either integer, float or dummy.

features = df_train.select_dtypes(include=['int64', 'float64', 'uint8', 'int8']).columns.tolist()

features.remove('log_revenue')

features_unseen = df_test.select_dtypes(include=['int64', 'float64', 'uint8', 'int8']).columns.tolist()

features_unseen.remove('log_revenue')



X, y = df_train[features], df_train['log_revenue']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)



print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)

#import libraries for light gbm and feature importance 

import lightgbm as lgb

import eli5



params = {'num_leaves': 15,

         'min_data_in_leaf': 25,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.8,

         "bagging_freq": 1,

         "bagging_fraction": 0.7,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1}



#build mogel 

lgb_model= lgb.LGBMRegressor(**params, n_estimators = 10000, nthread = 4, n_jobs = -1)

lgb_model.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=150)



eli5.show_weights(lgb_model, feature_filter=lambda x: x != '<BIAS>')

#make predictions 



#Prediction

y_pred=lgb_model.predict(X_test)
#look at rmse and R-Squared 

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt



rmse = sqrt(mean_squared_error(y_test, y_pred))





print ("R-Squared is:", metrics.r2_score(y_test, y_pred))

print ("The rmse is:", rmse)
#look at actual and predicted values 

compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

compare.head(5)
# look at actual and predicted values of first 50 entries in the dataset

compare1 = compare.head(50)

compare1.plot(kind='bar',figsize=(30,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
#make prediction on unseen test dataset

X_unseen=df_test[features_unseen]

prediction_unseen= lgb_model.predict(X_unseen)
prediction_unseen
df_submission['revenue'] = np.expm1(prediction_unseen)

df_submission.head(5)
df_submission[['id','revenue']].to_csv('submission_lgb.csv', index=False)