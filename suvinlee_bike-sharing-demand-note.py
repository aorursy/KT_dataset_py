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
train = pd.read_csv("../input/train.csv", parse_dates = ["datetime"])
test = pd.read_csv("../input/test.csv", parse_dates = ["datetime"])

train["year"] = train["datetime"].dt.year
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek

test["year"] = test["datetime"].dt.year
test["hour"] = test["datetime"].dt.hour
test["dayofweek"] = test["datetime"].dt.dayofweek

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
pd.options.display.max_rows = 200
submission["holiday"] = test["holiday"]
submission.loc[(submission["holiday"]==1)]
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
submission.drop("holiday",1,inplace=True)
submission.to_csv("allrf2.csv", index=False)
