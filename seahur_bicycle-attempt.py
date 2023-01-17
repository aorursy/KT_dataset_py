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
# read data (train, test) with pd.read_csv(directory)

train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

train.head(10)

#train.info()

#train.shape
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

test.head(10)
y = train['count']

# y의 편차가 매우 크다. log scaling으로 outlier를 제거한 듯한 효과를 내자.

# 회귀분석의 평가요소는 MSE(Mean Square Error) // 트리로 예측시 900을 100으로 예측하는 순간 800^2 = 6400만큼 패널티..

# 다른 속성들은 유지한 채, 스케일링으로 outlier를 제거한 효과를 위해 log sacling 하는 것.

y.sort_values()
# y분포를 확인해보자.

import matplotlib.pyplot as plt

import seaborn as sns

# 밑 그림 

wg, dh =  plt.subplots(2,1, figsize=(20,12))

# log scaling 전 분포 확인.

sns.distplot(y, ax=dh[0])

# log 씌웠더니 분포가 바뀔수록 좋은 것.

sns.distplot(np.log(y), ax=dh[1])

# 모델을 만들기 위해서는 가장 먼저, 모델에 돌릴 수 있는 상태로 만들어야 한다.

# y와 x를 별도로 저장. 그리고 train과 test 데이터의 변수가 같아야 한다.

# y = train['count']

# log scaling으로 y의 outlier를 제거한 듯한 효과.

y = np.log(train['count'])

y # 편차가 확실히 많이 줄었다. 분포가 조밀해짐.
############## 변수 추가를 위한 전처리.

# 날짜형식으로 바꾸는 3가지 서로 다른 방법.

train['datetime'] = train['datetime'].astype('datetime64')

# train['datetime'] = pd.to_datetime(train['datetime'])

# train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates = ['datetime'])

train.dtypes

test['datetime'] = test['datetime'].astype('datetime64')

# test['datetime'] = pd.to_datetime(test['datetime'])

# train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates = ['datetime'])

test.dtypes
train['year'] = train['datetime'].dt.year

train['weekday'] = train['datetime'].dt.weekday

train['hour'] = train['datetime'].dt.hour

train.head()
test['year'] = test['datetime'].dt.year

test['weekday'] = test['datetime'].dt.weekday

test['hour'] = test['datetime'].dt.hour

test.head()
# TIP) 날짜 형식으로 바꾸지 않고 바로 시간대를 추출하는 방법.

# time = train['datetime'].str.slice(11,13).astype(int)

# time.head()

# 모델 검증용으로 validate dataset 분리 7:3

#from sklearn.model_selection import train_test_split

#train_x, validate_x, train_y, validate_y  = train_test_split(train, y, test_size = 0.3,

                                                             #random_state = 777)
train = train.drop(['datetime', 'casual', 'registered', 'count'], 1)

test = test.drop('datetime', 1)

# 트리모델은 이해하기 쉽다. 그러나 진짜 이유는 카테고리형, 숫자형 데이터가 함께 존재할 때 랜포가 좋다.

#from sklearn.ensemble import RandomForestRegressor

# 모델 선언, 모델이 학습하는 방향을 설정해 줄 수 있다. n_estimator 너무 크면 과적합 그러나 랜포는 그렇게 심하진 않다 다른 모델에 비해.

# cpu를 전력으로 모두 쓰도록 안 해주면 1개만 씀. n_jobs=4 또는 -1로 하면 CPU를 모두 사용

# random_state는 set.seed()와 같음

#rf = RandomForestRegressor(n_estimators=100, n_jobs=-1,random_state=999)

#rf.fit(train, y)
#result = rf.predict(test)

                                                                               
#from lightgbm import LGBMRegressor

# boosting 기법은 hyper parameter 조정이 중요 (과소적합, 과대적합 예방)

#lgbm = LGBMRegressor()

#lgbm.fit(train, y)

#preds = lgbm.predict(test)
from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(train, y)

preds = predict(test)
sample = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")

sample.head()
# 처음에 np.log 적용한 상태로 train 시켰기 때문에, 예측값을 구할 때는 다시 exp 적용해줘야 한다.

sample['count'] = np.exp(preds)

sample.head()
sample.to_csv("sample.csv", index = False)

#################################################################################################

############################시각화 및 EDA를 통한 변수 INSIGHT얻는 과정###############################

y.sort_values()
train2 = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=['datetime'])

train2['year'] = train2['datetime'].dt.year

train2['month'] = train2['datetime'].dt.month

train2['day'] = train2['datetime'].dt.day

train2['weekday'] = train2['datetime'].dt.weekday

train2['hour'] = train2['datetime'].dt.hour



test2 = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv', parse_dates=['datetime'])

test2['day'] = test2['datetime'].dt.day



# 시간이라는 변수(정보)가 유의한지 보기 위해, 시간에 따른 자전거 수요 패턴 파악 가능

# mean은 위험할 수도 있다. outlier 때문에, median도 체크

#train2.groupby('hour')['count'].mean()

##### media을 보니 확실히 시간대별로 자전거 수요의 차이가 있음을 알 수 있다. 유의한 변수가 될 수 있음을 유추할 수 있다.

train2.groupby('hour')['count'].median()

a, b = plt.subplots(2,2,figsize=(20,12))

sns.boxplot(train2['year'], train2['count'], ax=b[0,1])

sns.boxplot(train2['month'], train2['count'], ax=b[1,1])

### day가 1~19일 밖에 없다.!

sns.boxplot(train2['day'], train2['count'], ax=b[0,0])

### 오후 시간에 왜 outlier가 많은가? 요일 (주중 5일/주말 2일) 주중 5일에 대표성이 된 것 임. 

### 주말 오후의 count가 많을 텐데 그것이 outlier로 잡히는 것.

sns.boxplot(train2['hour'], train2['count'], ax=b[1,0])

                                                  
np.unique(test2['day']) # test dataset: 20일 이후

#np.unique(train2['day']) # train dataset: 19일 까지

# 그래서 day 데이터는 y를 예측하는데 도움이 안된다.

# 데이터 수가 class별로 좀 부족.

train2['datetime'].dt.month.value_counts()