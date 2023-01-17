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
# 컬럼 뜻 확인하기
# 데이터양이 많은 경우, 모델을 빨리 돌릴 수 있는 것도 필요
# 정답값 확인하기 CATEGORY(정답값)
# 전처리 할때 데이터를 가져와야한다. 
# 1. 날짜 데이터 파싱해서 불러오기(index_col='Id':ID값을 인덱스로 쓰겠다.)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')
#데이터가 어떻게 생겼는지 보자
train.head()
#2. 문자형 데이터가 많다. 어떻게 처리할까(라벨 인코더)
# -> 문자를 숫자로 바꿔야겠다. 어떻게 바꿀까?

# 2.1 PdDistrict는 유니크한 카테고리가 10개 있다. 
train["PdDistrict"].nunique()
# 2.2 자동으로 텍스트를 숫자로 바꿔주기(레이블 인코더, 라벨 인코더)
# 라벨 인코더 불러오기
from sklearn.preprocessing import LabelEncoder
# 2.3 선언하기 
le1 = LabelEncoder()
# 2.4 모델링에서는 훈련/ 예측 이지만 여기서는 등록과 변환(le1 ->fit)
# 2.5 transform 사전(look up table)
# -> train(10개 데이터)에 사전을 등록
train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])
test['PdDistrict'] = le1.transform(test['PdDistrict'])

# 2.6 우리 모델이 학습할텐데, 숫자로 바꿔주면 데이터 사이에 거리가 생기진 않을까? 잘못된 정보
# -> 다행이도 괜찮다. -> 트리 모델은 각각 값이 질문을 해가면서 분류 작업을해나간다.
# -> 트리모델은 각각의 데이터가 다르기만 하면 잘 분류 된다(거리개념 적용x)
# -> 만약 회기 등이라면 가중치를 주며 학습할 수 있다. 
# 트리모델이 대표적:엔트로피가 낮다:무질서도 이기 때문에 데이터가 다 분류 된 상태라면 엔트로피가 낮아졌다.(만개를 분류할 때)
# 데이터에 가중치가 들어가는 모델일 경우(트리모델 외)= 원핫 인코딩, dummy화 하여 모델이 착각하지 않게 넣어줄 수 있다.
## 인코딩 기억해야할 것 2가지(라벨인코딩,원핫인코딩,pca차원축소를 통해 최대한 실제 정보를 보존하면서 칼럼수를 줄여버림)
# 또 인코딩 해줘야하는 것이 무엇일까?

#2.6.1 address도 인코딩
train.head()
#2.6.2 train, test가 개수가 다르다.
# train에는 없는데, test에만 있을 수 있다
# 이럴 경우 어떻게 하는게 좋을까?
train['Address'].nunique()
test['Address'].nunique()
#2.6.3 위의 값 진행하지 않고 아래 코드만 실행!
#train 과 test 유니크 값이 다르더라도 괜찮음.
#현업에서는 같이 등록하는 것 자체가 제한이 있다(test 확인x), 
#아래는 test 셋을 볼 수 있어서 대회 등에서 사용할 수 있다.
le3 = LabelEncoder()
le3.fit(list(train['Address']) + list(test['Address']))
train['Address'] = le3.transform(train['Address'])
test['Address'] = le3.transform(test['Address'])
#2.6.4. 라벨 인코딩 진행
# 기본 베이스라인 점수가 나옴
le2 = LabelEncoder()
y= le2.fit_transform(train['Category'])
#값 확인하기
train.head()
#트리 모델은 칼럼이 많을수록 좋다.
#트리모델은 학습할 때 알아서 피처를 선택한다. 
#중요한 칼럼이 있다면 가중치를 주면서 독점 학습
#반대로 안 중요한 것들은 학습 안함.
#모델에게 학습을 맡긴 후 이후에 빼도 된다.
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
#범죄 카테고리를 39개로 나눠야한다.
#범죄 트렌드를 추가해주면 모델 성능이 오를것이다.
#가령 사이버 범죄는 이전 보다 요즘 많다. 
#n_days 칼럼을 만들려면 train['date']코드를 만들어줘야한다.
#트레인 데이트에서 트레인 dt.date하면 연, 월,일 정보가 뽑힌다.
#(train['Date'] - train['Date'].min()) 
# : 어떤 범죄가 발생한 날짜 - 가장 최초 생긴 날짜
# = 3650 days 가 나온다. days를 삭제해줌(=3650)
# .apply(lambda x: x.days) : 사용자 지정 함수(한줄로 한방에)
# 요즘 많이 발생하는 범죄는 n.days가 크다.()

train['Date'] = (train['Dates'].dt.date)
train['n_days'] = (train['Date'] - train['Date'].min()).apply(lambda x: x.days)

test['Date'] = (test['Dates'].dt.date)
test['n_days'] = (test['Date'] - test['Date'].min()).apply(lambda x: x.days)
#00608문자를 숫자로만들어줌(원핫인코딩) -> 칼럼, 그릇이 너무 커진다.
#레이블인코딩 +랜덤포레스트 사용
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')

train['Date'] = pd.to_datetime(train['Dates'].dt.date)
train['n_days'] = (train['Date'] - train['Date'].min()).apply(lambda x: x.days)
train['Day'] = train['Dates'].dt.day
train['DayOfWeek'] = train['Dates'].dt.weekday
train['Month'] = train['Dates'].dt.month
train['Year'] = train['Dates'].dt.year
train['Hour'] = train['Dates'].dt.hour
train['Minute'] = train['Dates'].dt.minute
train['Block'] = train['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)
train['ST'] = train['Address'].str.contains('ST', case=False).apply(lambda x: 1 if x == True else 0)
train["X_Y"] = train["X"] - train["Y"]
train["XY"] = train["X"] + train["Y"]

test['Date'] = pd.to_datetime(test['Dates'].dt.date)
test['n_days'] = (test['Date'] - test['Date'].min()).apply(lambda x: x.days)
test['Day'] = test['Dates'].dt.day
test['DayOfWeek'] = test['Dates'].dt.weekday
test['Month'] = test['Dates'].dt.month
test['Year'] = test['Dates'].dt.year
test['Hour'] = test['Dates'].dt.hour
test['Minute'] = test['Dates'].dt.minute
test['Block'] = test['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)
test['ST'] = test['Address'].str.contains('ST', case=False).apply(lambda x: 1 if x == True else 0)
test["X_Y"] = test["X"] - test["Y"]
test["XY"] = test["X"] + test["Y"]

from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])
test['PdDistrict'] = le1.transform(test['PdDistrict'])

le2 = LabelEncoder()
y= le2.fit_transform(train['Category'])

le3 = LabelEncoder()
le3.fit(list(train['Address']) + list(test['Address']))
train['Address'] = le3.transform(train['Address'])
test['Address'] = le3.transform(test['Address'])

train.drop(['Dates','Date','Descript','Resolution', 'Category'], 1, inplace=True)
test.drop(['Dates','Date',], 1, inplace=True)
#모델링 과정 4가지 
#1.모델 가져오기 #모델 불러올 때 주의할 점:모든 모델은 분류/회기 모두 사용
#라이쥐비엔 
from lightgbm import LGBMClassifier

#2.모델 선언
model = LGBMClassifier()
#랜덤포레스트와, 부스팅 모델은 여러 나무를 만들고 각각의 나무가 전체 데이터 셋을 보면서 학습한다.
#랜덤포레스트 나무가100그루이면, 학습을 100번한다.학습을할 때 서로를 도와주지 않는다.
#각각의 나무들이(디시젼트리,의사결정)이 모여서 숲을 이룬다. 100그루이면 100번학습한다. 
#랜덤포레스트는 서로 돕지 않는다. 부스팅 모델은 서로서로 도와준다.
#부스팅 모델의 학습방법(1번나무에서 에러율이 10이 나왔다고 가정, 
#2번나무에서는 1번 나무의 틀ㄹ린것을 어떻게든 학습하려고 노력함,에러9)
#단, 이럴경우 오버피팅될 수 있다.(과접합문제) ->파라미터를 섬세하게 세팅
#모델의 성능을 극한으로 올리고 싶으면, 부스팅모델 사용

#<정말 중요한 하이퍼파라미터 설명>
#2.1 필수! colsample_bytree': 0.625 -> 과접합 문제 간접적으로 해결함(보통 0.7이고, 0.5~0.6까지 내려도 큰상관 없음)
#이 값을 설정안해주면, 모든 트리마다 같은 칼럼이 들어간다 
#나무마다 칼럼을 6개씩 다른 조합이 들어간다.62.5%이면 6개 
#왜이렇게 할까? 너무 같은 값, 100%로 들어가면 과접합
#트리모델은 탐욕적이라서 어떤 나무에 칼럼이 들어오면 중요한 칼럼이 있으면 가중치를 주면서 학습을 한다. 
#별로 안중요하면 칼럼을 버림=> 트리모델은 자동으로 피처를 선택(효율적임)
#4개의 칼럼이 있었음, 2번 칼럼이 가장 중요한 칼럼으로 과접합될 수 있어서, 퍼센테이지를 준다.

model = LGBMClassifier(colsample_bytree = 0.625)

#1번 나무에 1,2,3번 칼럼이 들어감 -> 2번 칼럼이 많이 들어감
#2번 나무에 1,3,4번 칼럼이 들어감 ->1번과 3번을 많이 포함, 
#별로 안중요한것도 학습되어 대처할 수 있으며, 과접합문제를 간접적으로 해결할 수 있다.
#각나무가 하나의 모델처럼, 앙상블과 비슷하다. (모델 성능이 어느정도 뒷받침,다양성확보)
#모델 속도도 40%가 높아진다.

#2.2 필수! subsample :0.8이 기본! 로우값 샘플링(데이터가 많으면 0.7까지는 낮출수있다.) 
#모델 학습속도 개선, 다양성확보,아웃라이어 
#데이터를 나무에 넣어줄 때 칼럼만 샘플링할수있을까?
#로우도 가능하다. 

model = LGBMClassifier(colsample_bytree = 0.625, subsample= 0.8)

#2.3 num_leaves:나뭇잎 개수:자식 노드 수:피쳐가 질문=피쳐(칼럼..), 무질서도가 낮아진다=앤트로피가 낮아진다.
#나뭇잎개수가 커지면, 오버피팅?언더피팅?->오버피팅 /정답 클래스가 39개이면(보이스피싱, 범죄) num_leaves도 최소 39개
#나뭇잎개수가 233개라면, 233만큼의 깊이, =?lgbm은 트리의 깊이 나뭇잎으로 설정한다.
#트리의 깊이가 7번,8번 정도 인거다. 2의7승=233개 
# num_leaves 기본은 31인데, 이거는 왜이렇게 많이 해줬을까? 
# 클래스가 39로 많아서, ex)주소데이터도 2만개로 복잡하다.
#우선 기본값으로 돌려보고 성능평가해본다음에 진행
model = LGBMClassifier(colsample_bytree = 0.625, subsample= 0.8, num_leaves = 233)

#2.4 n_estimators,learning_rate 짝꿍! 반비례,하나를 바꿔주면 같이 바꿔줘야함.
#n_estimators 나무 개수(보통 100개), 
#learning_rate(학습률 -> 학습속도, 하나의 나무에서 얼마나 학습을 할것인지!)
#러닝레이트가 크면 학습을 빨리한다. 최적의 코스트를 놓칠수있다. 오버슈팅 문제
#러닝레이트가 크다면 모델이 최적의 에러값으로 수렴하는데까지 걸리는 시간이 작다
#나무를 조금만 만들어줘도된다!!러닝레이트가 0.1 나무의갯수 150 
#/ 0.05 러닝레이트를 0.05로 낮춰주는 동시에 300개 0.025 나무의갯수 600개

model = LGBMClassifier(colsample_bytree = 0.625, subsample= 0.8, num_leaves = 233, learning_rate = 0.025, n_estimators = 600)
#학습
model.fit(train, y, categorical_feature=["PdDistrict", "DayOfWeek"])
#lgbm모델은 대용량 속도 가장 빠름! 문자열 처리 잘함 
#(Catboost은 문자열 가장 잘함, 최신 모델,대용량에는 약함)
#proba : 확률로 제출 ex) 정답값이 소매치기, 소매치기 확률 49%, 절도 51%이면 패널티를 적게 먹을 수있음
preds = model.predict_proba(test)

submission = pd.DataFrame(preds, columns=le2.inverse_transform(np.linspace(0, 38, 39, dtype='int16')), index=test.index)
submission.to_csv('LGBM_final.csv', index_label='Id')