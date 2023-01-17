import pandas as pd

%matplotlib inline

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm

import warnings; warnings.simplefilter('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=data=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

train.info()

Y1train=train['casual']

Y2train=train['registered']

Ytrain=train['count']

figure, axs = plt.subplots(nrows=3, ncols=2)



figure.set_size_inches(14,6)

sns.distplot(Ytrain, ax=axs[0][0], fit=norm)

sns.distplot(np.log(Ytrain+1), ax=axs[0][1], fit=norm)



sns.distplot(Y1train, ax=axs[1][0], fit=norm)

sns.distplot(np.log(Y1train+1), ax=axs[1][1], fit=norm)



sns.distplot(Y2train, ax=axs[2][0], fit=norm)

sns.distplot(np.log(Y2train+1), ax=axs[2][1], fit=norm)
feature_names=list(test)

train=train[feature_names]

all_data=pd.concat((train, test))

print(train.shape, test.shape, all_data.shape)

print(Ytrain)



all_data['datetime']=pd.to_datetime(all_data['datetime'])

all_data['year']=all_data['datetime'].dt.year

all_data['month']=all_data['datetime'].dt.month

all_data['day']=all_data['datetime'].dt.day

all_data['hour']=all_data['datetime'].dt.hour

all_data['dayofweek']=all_data['datetime'].dt.dayofweek

all_data=all_data.drop(columns='datetime')



all_data.loc[all_data['windspeed']==0, 'windspeed']=all_data['windspeed'].mean()

print(train.shape, test.shape, all_data.shape)
Xtrain=all_data[:len(train)]

Xtest=all_data[len(train):]

Xtrain.info()

#"""

import itertools

import copy

tmpXtrain = copy.deepcopy(Xtrain)

tmpXtest = copy.deepcopy(Xtest)



for cmb in itertools.combinations_with_replacement(list(Xtrain.keys()), 2):

    tmpXtrain["-".join(cmb)] = Xtrain[cmb[0]] * Xtrain[cmb[1]]

    tmpXtest["-".join(cmb)] = Xtest[cmb[0]] * Xtest[cmb[1]]

#"""
!pip install optuna
import optuna.integration.lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn import datasets



X_train, X_test, y_train, y_test = train_test_split(tmpXtrain, np.log1p(Y1train), test_size=0.1)



lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)



lgbm_params = {

    'objective': 'regression',

    'metric': 'rmse',

}

best_params, tuning_history = dict(), list()

booster_casual = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval,

                    verbose_eval=0,

                    best_params=best_params,

                   tuning_history=tuning_history)

print("Best Params:", best_params)

print("Tuning history:", tuning_history)

X_train, X_test, y_train, y_test = train_test_split(tmpXtrain, np.log1p(Y2train), test_size=0.1)



lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)



lgbm_params = {

    'objective': 'regression',

    'metric': 'rmse',

}

best_params, tuning_history = dict(), list()

booster_registered = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval,

                    verbose_eval=0,

                    best_params=best_params,

                   tuning_history=tuning_history)

print("Best Params:", best_params)

print("Tuning history:", tuning_history)
pred_casual = booster_casual.predict(tmpXtest, num_iteration=booster_casual.best_iteration)

pred_casual = np.expm1(pred_casual)



pred_registered = booster_registered.predict(tmpXtest, num_iteration=booster_registered.best_iteration)

pred_registered = np.expm1(pred_registered)



pred = pred_casual + pred_registered

pred[pred<0] = 0



submission = pd.DataFrame({'datetime': test.datetime, 'count': pred},

                          columns=['datetime', 'count'])

submission.to_csv("submission.csv", index=False)