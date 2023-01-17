# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import pandas as pd

import numpy as np

import math

from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



def evaluate(x_train, x_eval, y_train, y_eval, model):

    model.fit(x_train, y_train)

    y_predict = model.predict(x_eval)

    return math.sqrt(mean_squared_error(y_predict, y_eval))



# Read data

df_X_train = pd.read_csv('/kaggle/input/microsoft-data-science-capstone/Training_values.csv')

df_Y_train = pd.read_csv('/kaggle/input/microsoft-data-science-capstone/Training_labels.csv')

df_X_test  = pd.read_csv('/kaggle/input/microsoft-data-science-capstone/Test_values.csv')



df_X_train.describe()



# Clean missing data , Convert categorical features

ID = 'row_id'

target = 'heart_disease_mortality_per_100k'



X_train_1 = pd.get_dummies(df_X_train.drop(ID, axis=1).fillna(0)).values

Y_train_1 = df_Y_train[target].values

X_test_sub  = pd.get_dummies(df_X_test.drop(ID,axis=1).fillna(0)).values



# Split training data , Evaluate model , Tune hyper-parameters

X_train, X_test, y_train, y_test = train_test_split(X_train_1, Y_train_1, test_size=0.2, random_state=1)



from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



# param_test ={'num_leaves': sp_randint(6, 50), 

#              'min_child_samples': sp_randint(100, 500), 

#              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

#              'subsample': sp_uniform(loc=0.2, scale=0.8), 

#              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),

#              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

#              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],

#              'n_estimators': [1000, 1500, 2000, 2500, 3000, 3500, 4000],

#             'learning_rate' : [0.01, 0.03, 0.05, 0.07, 0.09],

#             'num_leaves': [10, 20, 30, 40, 50]}

# model = LGBMRegressor()

# gs = RandomizedSearchCV(

#     estimator=model, param_distributions=param_test,

#     scoring='neg_mean_absolute_error',

#     cv=3,

#     refit=True,

#     random_state=42,

#     verbose=True,

#     n_jobs=-1)

# gs.fit(X_train, y_train)

# print(gs.best_params_)



# model=LGBMRegressor(colsample_bytree=0.45860326840383037,leanring_rate=0.05, min_child_samples= 191, min_child_weight= 100,n_estimators=2000, num_leaves= 30, reg_alpha= 2, reg_lambda= 50, subsample= 0.3386917228062177)

# rmse = evaluate(X_train, X_test, y_train, y_test, model)



# print ('LGBMRegressor RMSE :', rmse)



model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=30, n_jobs=-1)

rmse = evaluate(X_train, X_test, y_train, y_test, model)

print ('LGBMRegressor RMSE :', rmse)



# Train model , Predict



Y_predict = model.predict(X_test_sub)

temp = pd.DataFrame(Y_predict)

temp.to_csv('submit.csv')
