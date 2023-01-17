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
import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sn

from scipy import stats

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from math import sqrt
train = pd.read_csv('../input/bike-share-demand/train.csv')

test = pd.read_csv('../input/bike-share-demand/test.csv')

train.head()
train["date"] = train.datetime.apply(lambda x : x.split()[0])

train["hour"] = train.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

train["year"] = train.datetime.apply(lambda x : x.split()[0].split("-")[0]).astype("int")

train["month"] = train.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

train.head()
test["date"] = test.datetime.apply(lambda x : x.split()[0])

test["hour"] = test.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

test["year"] = test.datetime.apply(lambda x : x.split()[0].split("-")[0]).astype("int")

test["month"] = test.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

test.head()
train.dtypes
print('train shape: ',train.shape)

print('test shape: ',test.shape)
train.isnull().sum()
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

sn.boxplot(data=train,y="cnt",orient="v",ax=axes[0][0])

sn.boxplot(data=train,y="cnt",x="season",orient="v",ax=axes[0][1])

sn.boxplot(data=train,y="cnt",x="hour",orient="v",ax=axes[1][0])

sn.boxplot(data=train,y="cnt",x="workingday",orient="v",ax=axes[1][1])



axes[0][0].set(ylabel='Count',title="Boxplot Count")

axes[0][1].set(xlabel='Season', ylabel='Count',title="Boxplot Season")

axes[1][0].set(xlabel='Hour', ylabel='Count',title="Boxplot Hour")

axes[1][1].set(xlabel='Workingday', ylabel='Count',title="Boxplot Workingday")
corrtrain = train[["temp","atemp","humidity","windspeed","cnt"]].corr()

mask = np.array(corrtrain)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sn.heatmap(corrtrain, mask=mask,vmax=.8, square=True,annot=True)
droptrain = ['datetime','casual','registered','cnt','date','atemp']

droptest = ['datetime','date','atemp']
training = train[pd.notnull(train['cnt'])].sort_values(by=["datetime"])

datetimecol = test["datetime"]

yLabels = [np.log(x) for x in train["cnt"]]
training  = training.drop(droptrain,axis=1)

testing  = test.drop(droptest,axis=1)

training.head()
X_train, X_val, y_train, y_val = train_test_split(training, yLabels, train_size=0.8, random_state=42)
models = [

           ['GradientBoostingRegressor: ', GradientBoostingRegressor()],

           ['RandomForestRegressor: ', RandomForestRegressor()],

           ['XGBRegressor: ', XGBRegressor()],

         ]
model_data = []

for name,curr_model in models :

    curr_model_data = {}

    curr_model.random_state = 78

    curr_model_data["Name"] = name

    curr_model.fit(X_train,y_train)

    y_pred = curr_model.predict(X_val)

    curr_model_data["Test_RMSLE_Score"] = sqrt(mean_squared_log_error(y_val,y_pred))

    model_data.append(curr_model_data)
df = pd.DataFrame(model_data)

df
model = CatBoostRegressor(eval_metric='MSLE')



def RMSLE(name, X_train, y_train, X_val, y_val):

    model.fit(X_train, y_train)

    y_predict = model.predict(X_val)

    print(name, sqrt(mean_squared_log_error(abs(y_predict), y_val)))



RMSLE('default',X_train, y_train, X_val, y_val)
from sklearn.model_selection import RandomizedSearchCV

params = {'depth':[6,7,8],

          'iterations':[1000,1350,1500],

          'learning_rate':[0.08,0.09,0.1], 

         }



rf_random = RandomizedSearchCV(

    estimator=model,

    param_distributions=params

)



rf_random.fit(X_train,y_train)

best_random = rf_random.best_estimator_



print(rf_random.best_params_)
# karena menggunakan RandomizedSearchCV maka setiap running akan menghasilkan parameter yang berbeda-beda. parameter terbaik yang didapatkan adalah sebagai berikut.

# iterations=1350

# learning_rate=0,08

# depth=6

modelfix = CatBoostRegressor(

    learning_rate= 0.08,

    iterations= 1350,

    depth= 6

)

modelfix.fit(X_train, y_train)

y_pred = modelfix.predict(X_val)

print(sqrt(mean_squared_log_error(abs(y_pred), y_val)))
prediksi = modelfix.predict(testing)

prediksi = [np.exp(x) for x in prediksi]
submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in (prediksi)]

    })
submission.to_csv('4899_amatir.csv', index=False)