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



#regression 문제 => 양을 맞추는 것
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv',parse_dates = ["datetime"])# 날짜 데이터 형식으로 바뀜 object가 아닌 날짜 형식 

train.head(30) # 자료 형태를 5개까지 볼 수 있음.()안에 숫자만큼.

train['hour']=train['datetime'].dt.hour   #hour, year 등 정보 추출 가능

train['year'] = train['datetime'].dt.year

train['dayofweek']=train['datetime'].dt.dayofweek #weekday

train['day']=train['datetime'].dt.day

train['month']=train['datetime'].dt.month

#train['week']=train['datetime'].dt.week



train.head(30)
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv',parse_dates = ["datetime"])# 이 자료의 자전거 수요량(count)을 예측해야 함.

test['hour'] = test['datetime'].dt.hour

test['year'] = test['datetime'].dt.year

test['dayofweek']=test['datetime'].dt.dayofweek

test['day']=test['datetime'].dt.day

test['month']=test['datetime'].dt.month

#test['week']=test['datetime'].dt.week

test.head(30)
weekday_df=train[train['workingday']==1] #조건 ==들어감. 모든 데이터에 접근해서 workingday가 1이면 True => True인 데이터만 가져옴. 

print(weekday_df.shape) #잘 추출했는지 확인



weekend_df=train[train['workingday']==0]

weekend_df.shape
import matplotlib.pyplot as plt #밑그림을 그릴 때 matplotlib 보통 사용

import seaborn as sns #실제 그래프를 그리는 라이브러리 

a,b = plt.subplots(1,1,figsize=(20,12)) #행,열,전체 사이즈

sns.boxplot(train['hour'],train['count']) #boxplot => 숫자형(count), 카테고리형(hour)을 그래프로 표현하고 싶을 때 

#박스 내부 선 => 중앙값

#박스 내부 맨 위 선=> 상위 25%

#박스 외부 맨 위 선 => 최댓값

#박스 외부 맨 위 선 위의 점 => 이상치. 정상적인 데이터라고 인식하지 않은 것. 표준편차 바깥의 데이터들

# 10시 ~15시 보면 이상치가 많음. => 일주일 중 5일에 해당하는 데이터들은 박스로 잡히고, 이상치로 잡힌 데이터는 주말인 것. => 평일 주말 나눠서 그래프 그려줘야함.=> 이렇게 그리면 해석이 잘 안됨.

a,b = plt.subplots(1,1,figsize=(20,12))

sns.distplot(train['count'])#한 변수의 분포 그래프

#알 수 있는 패턴

# 회귀 문제,

#1.데이터가 한쪽으로 쏠려 있다. 

#2.이상치가 꽤 있다.(꼬리가 길게 보여짐)



# y(train의 count)에 로그를 취해줌

# train 셋에서의 이상치들이 test셋에서도 나온다는 보장이 없음. 

#이상치를 제거하는 방법은? => 굉장히 위험. 모델의 전체적인 분포를 잡는 착한 이상치들까지 제거할 수 있음. 오히려 전반적인 모델의 성능이 낮아질 수 있음. 

#=> 이상치를 제거하지 않고도 제거하는 듯한 효과를 볼 수 있는 방법 => y값에 로그를 취하는 것.

#=> 어느정도 작아진 값들을 갖고 학습을 하기 때문에 점수가 확 떨어지지 않음. 



a,b = plt.subplots(1,1,figsize=(20,12)) 

sns.boxplot(train['month'],train['count']) #tree모델은 중요한 column만 학습을 해주기 때문에 1차적으로 알아서 중요도가 낮은 column들은 걸러줌. 여기서는 day column이 중요하다 생각했기 떄문에 학습을 해버림. 
#month => train 셋에 들어있는 count의 분포와 test셋에 들어있는 count의 분포가 다르기 때문에. 전자는 월초중반, 

# 후자는 월말. 오히려 train 셋을 학습시키는게

# test셋에 부정적 효과. 예측하려는 것은 20~31일의 값. 
a,b = plt.subplots(1,1,figsize=(20,12)) 

sns.boxplot(weekday_df['hour'],weekday_df['count']) 
a,b = plt.subplots(1,1,figsize=(20,12)) 

sns.boxplot(weekend_df['hour'],weekend_df['count']) 
figure, (a,b,c,d,e,f) = plt.subplots(nrows=6)

figure.set_size_inches(18,25)



sns.pointplot(train['hour'],train['count'], ax = a)

sns.pointplot(train['hour'],train['count'],hue = train['workingday'], ax = b)

sns.pointplot(train['hour'],train['count'],hue = train['holiday'], ax = c)

sns.pointplot(train['hour'],train['count'],hue = train['dayofweek'], ax = d)

sns.pointplot(train['hour'],train['count'],hue = train['season'], ax = e)

sns.pointplot(train['hour'],train['count'],hue = train['weather'], ax = f)
print(train.groupby('year')['count'].mean()) #통계랑을 보는 기본적인 방법. 

train.groupby('year')['count'].median() #극단치.이상치 => 중간값 봐보자
train_2011=train[train['year']==2011] #2011년만 추출.

train_2011.groupby('month')['count'].mean()

a,b=plt.subplots(1,1,figsize=(20,12))

sns.boxplot(train_2011['month'],train['count'])
print(train.groupby('dayofweek')['count'].mean()) # 0~6 => 월~일

train.groupby('holiday')['count'].mean() #요일을 세분화하면 더 좋지 않을까?
train['dayofweek'].value_counts() #하나의 카테고리에 몇개의 데이터가 들어가있는지 확인. 뭔가 데이터가 애매할 때 이렇게 확인하는 방법도 있음. 유의미한지 확인하는 방법 중 하나. 
train.dtypes #datetime의 자료 형태는 모두 object. 숫자 형태가 아님. 숫자형태만 컴퓨터가 인지하고 머신러닝 돌릴 수 있음.
train2 = train.drop(['datetime','casual','registered','count','month','day'],axis=1) #숫자 형태 아닌 datetime빼주고, test에는 없는 3개의 항목들을 빼주는 작업. 그 수를 맞춰야 함. 

# train2라는 변수를 만들어 4개의 항목들을 빼준 새로운 데이터셋을 저장해둠. axis=0 -> row를 의미. axis=1 -> column 의미. 

train2.head()
test2 = test.drop(['datetime','month','day'],axis=1) # test에서도 숫자 형태가 아닌 datetime 열 삭제 

test2.head()
# #모델 불러오기

# from sklearn.ensemble import RandomForestRegressor 



# #모델 선언하기

# rf = RandomForestRegressor(n_estimators=100,random_state=1,n_jobs=4) #옵션을 넣어주자 / /



#랜덤포레스트;기본값 10->충분한 학습을 하지 못함 => 나무 100그루 심어주는 옵션 / 다른 옵션도 들어갈 수 있음 / 

#random_state =>값 고정 / n_jobs =4가 최댓값,빨리 돌아감



# #왜 100으로 설정? => 나무의 개수를 늘린다고 점수가 계속 좋아지는건 아님. 그만큼 train set을 

#집중적으로 들어가 학습을 많이 한다는 뜻. test set에 적용할 때 a=b여야 한다는 고집이 생겨서 

#조금만 달라도 다른 예측을 해버릴 수 있음



# #학습과정

# rf.fit(train2,np.log(train['count']))

# #예측하기

# result = rf.predict(test2)

# test['count']= result #test의 count 열을 만들어 result 값들을 넣어주는 작업. 

# test.head(10)



# # column 수가 많아지고, 데이터가 복잡해졌으면 이에 따라 모델 또한 발전시켜줘야함. 충분히 학습할 수 있게. 

# # 그렇지 않을 때 문제점 발생

# #1. 점수가 더 오를 수 있는데 그러지 못함.

# #2. 오히려 점수가 안좋아지는 경우도 있음. 





#기본 나무 개수가 100으로 설정. 이 데이터셋의 복잡도에 lgbm 기본 옵션값들이 잘 맞아떨어져서 운이 좋게 점수 잘 나온 것. 

#부스팅모델 lgbm

# from lightgbm import LGBMRegressor

# lgb=LGBMRegressor()

# lgb.fit(train2,np.log(train['count']))

# result=lgb.predict(test2)

# test['count']=result

# test.head()



#이 대회는 lgbm이 xgboost보다 성적이 잘나오는 대회라서? => X. 성능이 비슷하게 나옴. 



#부스팅모델2 xgboost =>랜덤포레스트처럼 옵션 넣어주면 됨. 나무는 기본 100으로 설정됨.

# *옵션값들 설정시 주의. 이 대회에 안맞는 것. 옵션값들을 상세하게 설정해줘야함.



# # 하이퍼파라미터***

# tree계열 모델=>

# tree의 깊이. 학습을 얼마나 더 깊이 할 수 있는가 => max_depth

#xgb=> max_depth기본 3 설정돼있음. 질문 자체를 많이 못하면 학습을 잘 못하게됨. 





from xgboost import XGBRegressor

xgb=XGBRegressor(nthread=4,max_depth=5) # 카톡 사진 참고. 사진에 나온 것이 나무 하나. nthread=>빨리 돌아가게 하는 하이퍼파라미터. n_jobs와 같은 옵션. CPU를 다 써서.

xgb.fit(train2,np.log(train['count']))

result=xgb.predict(test2)
Sub = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')

Sub.head()
Sub['count'] = np.exp(result) #다시 원상복귀 #지수화

Sub.head()
Sub.to_csv('20191231.csv',index=False) # index=False 안해주면 index=True가 되어 열하나 더 생김. 제출 시 column이 2개여야함. 