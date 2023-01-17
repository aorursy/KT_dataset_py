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
train = pd.read_csv("../input/train.csv")
train.head()
train.info()
train.dtypes
train = pd.read_csv("../input/train.csv", parse_dates = ["datetime"])
test = pd.read_csv("../input/test.csv", parse_dates = ["datetime"])
train.dtypes
train["datetime"] = train["datetime"].astype("datetime64")
train["datetime"] = pd.to_datetime(train["datetime"])
train["datetime"].dt.hour
train.head()
train["year"] = train["datetime"].dt.year
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek

test["year"] = test["datetime"].dt.year
test["hour"] = test["datetime"].dt.hour
test["dayofweek"] = test["datetime"].dt.dayofweek
train.head()
train["datetime"].dt.day.value_counts()
train.head()
# 머신러닝 모델링을 할 때 반드시 해야하는 2가지 전처리 기법
# 1. 칼럼의 갯수를, 칼럼의 조합을 반드시 맞춰주어야 한다! 칼럼순서는 상관없음~
# 2. y 종속변수(타겟) 다른 변수에 미리 넣어두기
y_train = train["count"]
test.head()
train.shape
train = train.drop(["datetime", "windspeed", "casual", "registered", "count"], 1) # axis=1 (column 제거), 행 제거할 때는 0 또는 빈칸
test.drop(["datetime", "windspeed"], 1, inplace=True)
train.head()
test.head()
y_train = np.log1p(y_train) # log p+1을 계산
# 랜덤포레스트 모델링
# 모델링 4가지 과정
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(train, y_train)
preds = rf.predict(test)
submission=pd.read_csv("../input/sampleSubmission.csv")
submission["count"] = preds
submission.head()
submission["count"] = np.expm1(preds)
submission.head()
submission.to_csv("allrf.csv", index=False)