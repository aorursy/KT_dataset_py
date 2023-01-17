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
#1.디스플레이 옵션 값 넣기, row개수 정해줌

pd.options.display.max_rows = 200

#2.submisson 파일에 holiday 값을 넣어줌

submission["holiday"] = test["holiday"]

#3.loc,iloc 함수: 어떤 데이터에 접근해서, holiday==1 값 가져오기

submission.loc[(submission["holiday"]==1)]

# 아래 값 이상한점은?

# 3.1 분산이 크다.

# 3.2 train count 값이 대부분 왼쪽에 쏠려 0~100사이에 있었다.

#     그렇다면, test 예측 값도 0~100사이여야하는데,

#     아래 데이터에서는 200~까지 있다.

# 3.3 우리가 홀리데이=1인경우를 뽑았기 때문에 높게 형성될 수 있다.

#     => 잘못 예측하고 있다. 

# 3.4 train set(1~19일) 과 test set 의 날짜 데이터셋이 맞지 않는다.

#     => 공휴일이 다름

## 모델 성능을 위해 특정 날을 처리해줘야한다.

## 도메인 지식= 전문 지식 
#loc:어떤 칼럼에 접근할 때, 칼럼에 이름을 넣어줘서 접근한다.

#iloc: 이름이 아닌 칼럼의 순서로 접근한다.

#왼쪽이 행, 오른쪽이 열(1)

#정답값(count)를 50% 낮게 맞춰줌 

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

submission.drop("holiday",1,inplace=True)

submission.to_csv("allrf2.csv", index=False)