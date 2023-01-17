# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#1.Light GBM:라이트 그래디언트 부스팅 모델 사용 이유 /LGBM의 궁극적인 역할은 문자를 숫자로 바꿔준다.
#-문자열 데이터가 많은 경우 LGBM 사용
#-문자열 잘 처리하는 모델은 캣부스트가 가장 잘하고, LGBM이 두번째이지만, LGBM은 대용량 처리에 용이함
#-데이터에 적절한 좋은 성능의 모델을 사용 

#2.데이터불러오기
train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')
train.head()
#3.1. 날자 데이터 전처리 (dt :datetime) 
train['Day'] = train['Dates'].dt.day
train['DayOfWeek'] = train['Dates'].dt.weekday
train['Month'] = train['Dates'].dt.month
train['Year'] = train['Dates'].dt.year
train['Hour'] = train['Dates'].dt.hour
train['Minute'] = train['Dates'].dt.minute

test['Day'] = test['Dates'].dt.day
test['DayOfWeek'] = test['Dates'].dt.weekday
test['Month'] = test['Dates'].dt.month
test['Year'] = test['Dates'].dt.year
test['Hour'] = test['Dates'].dt.hour
test['Minute'] = test['Dates'].dt.minute
train.dtypes
#3.2 dates칼럼에서 칼럼하나 추가
#- 범죄 트렌드가 있을것, 예전에 많이 발생했을 범죄, 최근에 많이 발생한 범죄
#train['Date'] = train['Dates'].dt.date
#각 데이터마다 범죄 발생시간이 있고, 우리 데이터셋이 작성 시점부터 얼마나 지났는지 칼럼 추가
#train['Date'] - train['Date'].min()
#문자형 데이터여서(days) 숫자형으로 바꿔줘야함, apply 함수 적용(인덱스가 있는 경우 일괄처리,모든 데이터에 한번에 접근,pandas내장함수)
#(train['Date'] - train['Date'].min()).apply()
#apply 람다 함수: 무엇을 적용할거냐? x는 각각 데이터를 의미하고, x.days라는 함수를 사용할것이다.
(train['Date'] - train['Date'].min()).apply(lambda x: x.days)
train['Date'] = pd.to_datetime(train['Dates'].dt.date)
train['n_days'] = (train['Date'] - train['Date'].min()).apply(lambda x: x.days)

test['Date'] = pd.to_datetime(test['Dates'].dt.date)
test['n_days'] = (test['Date'] - test['Date'].min()).apply(lambda x: x.days)
#최근 범죄는 n_days가 크다, 오래된 범죄는 n-days가 작다.
#3.3 주소 정보에서 추가
#라벨링할 경우의 문제점: 
#1) 23000개의 고유값이 들어가 있는 칼럼: 데이터 길이가 크다.
#2) 주소의 정보값이 뭉개진다.(도/시/동 특성별 다를 수 있다.도메인 지식)

#train['Address'].str.contains('block', case=False)
#str은 특수문자, 숫자등이 들어간 것들 모두 문자로 바꿔줌!
#contains(뒤의 내용을 포함하고있냐 물어보는 함수, 강남이란 정보를 포함하고 있으면 트루,case=false는 영어 대소문자 구분하지 않겠다)

train['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)
#if x가 true이면, 그 데이터를 1로 바꿔주겠다. 아니면 0이다.

train['Block'] = train['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)
train['ST'] = train['Address'].str.contains('ST', case=False).apply(lambda x: 1 if x == True else 0)

test['Block'] = test['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)
test['ST'] = test['Address'].str.contains('ST', case=False).apply(lambda x: 1 if x == True else 0)

#블락이랑 스트릿을 넣어줘서 좋은 모델 성능 달성, 놓치는 부분 없을까?
#중요한 정보를 인간이 판단해서 추가해 주면, 더 중요한 정보를 놓칠수도 있다. 혹은 더 중요하지 않을 수 있다.
#주소값이 중요한지 안한지 딥러닝으로 판단 어떤 단어와 유사한지 안한지 모델이 알아서 판단해서 피처 추가(임베딩 방법론)
#3.4 위도 경도를 더하고 빼줬다(사칙연산). 이게 무슨짓이지? 의미가 없어 보이는데, 왜 성능이 높아질까=>데이터거리개념생김(각도,방향)
#의미가 있을만한것을 더하고 빼주면 더 도움이 된다. 우리는 모르지만 실험을 해봤을 때 의미가 있을수 있다. 
#위도 경도     
#1       2
#2       3
#위도-경도 
#1
#1
#피처엔지니어링에서 고급 내용
#3.4.1 우리가 사용하는 트리모델. 트리모델의 단점 2개
#- 거리기반의 학습을 잘 못한다.(이런 부분을 앙상블할때 사용)
#- 피처간의 인턴렉션이 없다.(독립적으로만 학습을 잘한다.인공신경망은 같이 학습을 잘한다.몸무게,키)
train["X_Y"] = train["X"] - train["Y"]
train["XY"] = train["X"] + train["Y"]

test["X_Y"] = test["X"] - test["Y"]
test["XY"] = test["X"] + test["Y"]

