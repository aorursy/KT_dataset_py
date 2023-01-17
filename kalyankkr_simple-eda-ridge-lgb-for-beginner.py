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

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import Ridge, Lasso,ElasticNet

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from catboost import Pool, CatBoostRegressor, cv

import xgboost as xgb

import lightgbm as lgb

train = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")

test = pd.read_csv("/kaggle/input/liverpool-ion-switching/test.csv")
train.shape
train.head()
train["open_channels"].value_counts().plot(kind = "bar")

plt.xlabel('open Channels')

plt.ylabel('count')

plt.title("Distribution of Channels")

plt.show()
train["open_channels"].value_counts(normalize=True)
print(f"We have {train.shape[0]//500000} Batches in the training dataset")
#for every 500000 values we conconstrct a hist plot



fig, axes = plt.subplots(4,3, figsize=(10, 10))

fig.subplots_adjust(hspace=0.5)

fig.suptitle('Distributions of Targets for batches')

for ax, i in zip(axes.flatten(),range(0,10)):

    

    data = train.iloc[(i * 500000):((i+1) * 500000 + 1)]['open_channels']

    sns.countplot(data, ax= ax)

    ax.set(title=f"Batch-{i}".upper())
#plot of the signal

plt.figure(figsize=(20, 8))

train.signal.plot()

plt.title("Train data")
trnX = train[['signal']].values

trnY = train['open_channels'].values
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.5)

model.fit(trnX, trnY)
ridge_predictions = model.predict(trnX)

ridge_arounded = np.rint(ridge_predictions).astype(int)
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

testX = test[['signal']].values

ridge_prediction = model.predict(testX)

ridge_prediction = np.rint(ridge_prediction).astype(int)

model_lgb = lgb.LGBMRegressor(objective='regression',

                              num_leaves=4,

                              learning_rate=0.05, 

                              n_estimators=1080,

                              max_bin=75, 

                              bagging_fraction=0.80,

                              bagging_freq=5, 

                              feature_fraction=0.232,

                              feature_fraction_seed=9, 

                              bagging_seed=9,

                              min_data_in_leaf=6, 

                              min_sum_hessian_in_leaf=11)
model_lgb.fit(trnX, trnY)
lgb_train_pred_trn = model_lgb.predict(trnX)

lgb_train_pred = model_lgb.predict(testX)

lgb_pred = np.rint(lgb_train_pred).astype(int)
#ensemble

ensemble = ridge_prediction*0.35 + lgb_pred*0.65

submission['open_channels'] = ensemble

submission[submission['open_channels']<0]['open_channels'] = 0

submission.head()

submission.to_csv("submission.csv", index=False)