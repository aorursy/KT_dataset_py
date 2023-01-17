# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import sklearn.metrics as metrics

import math
path="../input/insurance/insurance.csv"

df=pd.read_csv(path)

df.head(10)
object_columns_df = df.select_dtypes(include=['object'])

numerical_columns_df =df.select_dtypes(exclude=['object'])
object_columns_df.head()
numerical_columns_df.head()
#Number of null values in each feature

null_counts = object_columns_df.isnull().sum()

print("Number of null values in each column:\n{}".format(null_counts))
#Number of null values in each feature

null_counts = numerical_columns_df.isnull().sum()

print("Number of null values in each column:\n{}".format(null_counts))
df.region.value_counts()
df.sex.value_counts()
cop = df

cop.head()
bin_map ={"female":1,"male":2 ,"southeast":1,"southwest":2,"northwest":3,"northeast":4,"yes": 1,"no":2}

cop['sex'] = cop['sex'].map(bin_map)

cop['smoker'] = cop['smoker'].map(bin_map)

cop["region"]= cop["region"].map(bin_map)

           
cop.head()
features = ["age","sex","bmi","children","smoker","region"]

X = cop[features]

target = cop["charges"]
x_train,x_test,y_train,y_test = train_test_split(X,target,random_state=0)


forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(x_train, y_train)

preds = forest_model.predict(x_test)

print(mean_absolute_error(y_test, preds))

print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, preds))))
xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=0,

             importance_type='gain', learning_rate=0.01, max_delta_step=0,

             max_depth=4, min_child_weight=1.5, n_estimators=2400,

             n_jobs=1, nthread=None, objective='reg:linear',

             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 

             silent=None, subsample=0.8, verbosity=1)
#Fitting

xgb.fit(x_train, y_train)

predict1 = xgb.predict(x_test)

print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict1))))
x_ax = range(len(y_test))

plt.figure(figsize=(25,6))

plt.plot(x_ax, y_test, label="original")

plt.plot(x_ax, predict1, label="predicted")

plt.title("Medical cost Prediction by XGB Regressor")

plt.legend()

plt.show()
x_ax = range(len(y_test))

plt.figure(figsize=(20,6))

plt.plot(x_ax, y_test, label="original")

plt.plot(x_ax, preds, label="predicted")

plt.title("Medical cost Prediction by RandomForest")

plt.legend()

plt.show()
submission = pd.DataFrame({

        

        "charges": preds

    })

submission.to_csv('submission.csv', index=False)
submission.head(100)