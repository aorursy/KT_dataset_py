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
train_data = pd.read_csv('/kaggle/input/jejubusdacon/train.csv')
test_data = pd.read_csv('/kaggle/input/jejubusdacon/test.csv')
train_data.head()
train_data.info()
import numpy as np # 행렬 연산 / 데이터 핸들링
import pandas as pd # 데이터 분석
import matplotlib.pyplot as plt # 그래프 시각화
import seaborn as sns # 그래프 시각화
from xgboost import XGBRegressor # XGBoost Regressor 모델
from sklearn.model_selection import KFold # K-validation
from sklearn.metrics import accuracy_score # 정확도 측정 함수
from sklearn.preprocessing import LabelEncoder # 라벨 인코더
# !pip install 패키지 이름 
# !pip install -U finance-datareader
idx
idx = (train_data['next_arrive_time'] <= 700)
train_data = train_data.loc[idx,:]
train_data
station_encoder = LabelEncoder() # 인코더 생성
# _station = list(train_data['now_station'].values) + list(train_data['next_station'].values) # train_data 의 모든 정류장 이름
_station = list(train_data['now_station']) + list(train_data['next_station']) # train_data 의 모든 정류장 이름
station_set = set(_station)
print(len(station_set))
# len([[1,2,3,4,5,6],[1,2,3,4,5,6]])
# len('Hello World')

station_encoder.fit(list(station_set)) # 인코딩
station_encoder

train_data['now_station'] = station_encoder.transform(train_data['now_station'])
train_data['next_station'] = station_encoder.transform(train_data['next_station'])
test_data['now_station'] = station_encoder.transform(test_data['now_station'])
test_data['next_station'] = station_encoder.transform(test_data['next_station'])
train_data.head()
times_ = train_data['now_arrive_time'] # 
times_.hist()
target_ = train_data['next_arrive_time']
target_.hist(bins=50)

target_ = train_data['distance']
target_.hist(bins=50)

train_data['date'] = pd.to_datetime(train_data['date']) # date 값을 datetime으로
train_data['weekday'] = train_data['date'].dt.weekday  # Monday 0, Sunday 6
train_data['weekday'] = train_data['weekday'].apply(lambda x: 0 if x <= 5 else 1) 
# 0 ~ 5 는 월요일 ~ 금요일이므로 평일이면 0, 주말이면 1을 설정하였다
train_data['weekday'].unique()
train_data = pd.get_dummies(train_data, columns=['weekday']) # 평일/주말에 대해 One-hot Encoding
train_data = train_data.drop('date', axis=1) # 필요없는 date 칼럼을 drop
train_data.head()
test_data['date'] = pd.to_datetime(test_data['date'])
test_data['weekday'] = test_data['date'].dt.weekday  # Monday 0, Sunday 6
test_data['weekday'] = test_data['weekday'].apply(lambda x: 0 if x <= 5 else 1)
test_data = pd.get_dummies(test_data, columns=['weekday'])

test_data = test_data.drop('date', axis=1)
test_data.head()
train_data['time_group']='group' #time_group 변수를 미리 생성

train_data.loc[ (train_data['now_arrive_time']>='05시') & (train_data['now_arrive_time']<'12시') ,['time_group'] ]= 'morning' # 05~11시
train_data.loc[ (train_data['now_arrive_time']>='12시') & (train_data['now_arrive_time']<'18시') ,['time_group'] ]= 'afternoon' #12~17시
train_data.loc[ (train_data['now_arrive_time']>='18시') | (train_data['now_arrive_time']=='00시'),['time_group'] ]= 'evening' #18~00시

train_data = pd.get_dummies(train_data,columns=['time_group']) # 원 핫 인코딩을 수행
train_data = train_data.drop('now_arrive_time', axis=1) # 필요없는 now_arrive_time drop
train_data.head()
test_data['time_group']='group'

test_data.loc[ (test_data['now_arrive_time']>='05시') & (test_data['now_arrive_time']<'12시') ,['time_group'] ]= 'morning' # 05~11시
test_data.loc[ (test_data['now_arrive_time']>='12시') & (test_data['now_arrive_time']<'18시') ,['time_group'] ]= 'afternoon' #12~17시
test_data.loc[ (test_data['now_arrive_time']>='18시') | (test_data['now_arrive_time']=='00시'),['time_group'] ]= 'evening' #18~00시

test_data = pd.get_dummies(test_data,columns=['time_group'])
test_data = test_data.drop('now_arrive_time', axis=1)
test_data.head()
train_data = train_data.drop(['id', 'route_nm', 'next_latitude', 'next_longitude', 
                              'now_latitude', 'now_longitude'], axis=1)
train_data.head()
test_data = test_data.drop(['route_nm', 'next_latitude', 'next_longitude', 
                              'now_latitude', 'now_longitude'], axis=1)
test_data.head()
input_var = list(train_data.columns) 
input_var.remove('next_arrive_time')

Xtrain = train_data[input_var] # 학습 데이터 선택
Ytrain = train_data['next_arrive_time'] # target 값인 Y 데이터 선택

Xtest = test_data[input_var] # 시험 데이터도 선택
model = XGBRegressor(random_state=110, verbosity=0, nthread=23, n_estimators=980, max_depth=4)
kfold = KFold(n_splits=8, shuffle=True, random_state=777)
n_iter = 0
cv_score = []

def rmse(target, pred):
    return np.sqrt(np.sum(np.power(target - pred, 2)) / np.size(pred))
# i = 0 
for train_index, test_index in kfold.split(Xtrain, Ytrain):
    # K Fold가 적용된 train, test 데이터를 불러온다
    X_train, X_test = Xtrain.iloc[train_index,:], Xtrain.iloc[test_index, :]
    Y_train, Y_test = Ytrain.iloc[train_index], Ytrain.iloc[test_index]
    
    # 모델 학습과 예측 수행
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print(pred)
    
    # 정확도 RMSE 계산
    n_iter += 1
    score = rmse(Y_test, pred)
    print(score)
    cv_score.append(score)
#     i += 1
#     print(i)
print('\n교차 검증별 RMSE :', np.round(cv_score, 4))
print('평균 검증 RMSE :', np.mean(cv_score))
