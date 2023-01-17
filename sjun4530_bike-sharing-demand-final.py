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
train = pd.read_csv("../input/train.csv", parse_dates = ["datetime"])

test = pd.read_csv("../input/test.csv", parse_dates = ["datetime"])



train["year"] = train["datetime"].dt.year

train["hour"] = train["datetime"].dt.hour

train["dayofweek"] = train["datetime"].dt.dayofweek



test["year"] = test["datetime"].dt.year

test["hour"] = test["datetime"].dt.hour

test["dayofweek"] = test["datetime"].dt.dayofweek
train.info() # train.shape, train.isnull().sum(), train.dtypes 한눈에 확인 가능
train.describe() # 치우침, 아웃라이어, 특성마다 숫자형인지 범주형인지 확인
train['temp'].value_counts().sort_index() # 숫자의 의미가 어느정도 있는 범주형 특성(binning)
import seaborn as sns

import matplotlib.pylab as plt



_, axes = plt.subplots(1,1, figsize = (20,12))

sns.boxplot(x=train["hour"], y=train["count"])
fig, axes = plt.subplots(3,1, figsize = (20,12))



sns.countplot(train["season"], ax = axes[0], palette="Set1")

sns.countplot(train["weather"], ax = axes[1], palette="Set1")

sns.countplot(train["windspeed"], ax = axes[2])
fig, axes = plt.subplots(3,1, figsize = (20,12))



sns.countplot(train["season"], ax = axes[0], palette="Set1")

sns.countplot(train["weather"], ax = axes[1], palette="Set1")

sns.countplot(train["windspeed"], ax = axes[2])

plt.xticks(rotation = 60, )
y_casual = np.log1p(train.casual)

y_registered = np.log1p(train.registered)

#y_train = np.log1p(train["count"])



train.drop(["datetime", "windspeed", "casual", "registered", "count"], 1, inplace=True)

test.drop(["datetime", "windspeed", ], 1, inplace=True)
import lightgbm as lgb

hyperparameters = { 'colsample_bytree': 0.725,  'learning_rate': 0.013,

                    'num_leaves': 56, 'reg_alpha': 0.754, 'reg_lambda': 0.071, 

                    'subsample': 0.523, 'n_estimators': 1093}

model = lgb.LGBMRegressor(**hyperparameters)

model.fit(train, y_casual)

preds1 = model.predict(test)



hyperparameters = { 'colsample_bytree': 0.639,  'learning_rate': 0.011,

                    'num_leaves': 30, 'reg_alpha': 0.351, 'reg_lambda': 0.587,

                   'subsample': 0.916, 'n_estimators': 2166}

model = lgb.LGBMRegressor(**hyperparameters, )

model.fit(train, y_registered)

preds2 = model.predict(test)



submission=pd.read_csv("../input/sampleSubmission.csv")

submission["count"] = np.expm1(preds1) + np.expm1(preds2)

#submission.to_csv("allrf.csv", index=False)
#미국 현충일

submission.iloc[1258:1269, 1]= submission.iloc[1258:1269, 1]*0.5

submission.iloc[4492:4515, 1]= submission.iloc[4492:4515, 1]*0.5

#크리스마스 이브

submission.iloc[6308:6330, 1]= submission.iloc[6308:6330, 1]*0.5

submission.iloc[3041:3063, 1]= submission.iloc[3041:3063, 1]*0.5

#크리스마스

submission.iloc[6332:6354, 1]= submission.iloc[6332:6354, 1]*0.5

submission.iloc[3065:3087, 1]= submission.iloc[3065:3087, 1]*0.5

#추수감사절

submission.iloc[5992:6015, 1]= submission.iloc[5992:6015, 1]*0.5

submission.iloc[2771:2794, 1]= submission.iloc[2771:2794, 1]*0.5
submission.iloc[6332:6354]
submission.to_csv("lgb.csv", index=False)