"""

Bike Sharing Demand 진행방향

1) 훈련, 테스트 데이터셋의 형태 및 컬럼의 속성 데이터 값 파악

2) 데이터 전처리 및 시각화

3) 회귀모델 적용

4) 결론 도출



함수 사용시 꿀팁



함수를 적용 시 내부 파라미터들을 모를 때 Anaconda Prompt or Windows PowerShell을 활용하여 내부의 REPL python 명령창에서

ex)) pandas.to_numeric() 함수의 내부 parameter를 알고 싶다면

help(pandas.to_numeric)하게 되면, 함수의 사용법 등 문서를 열람할 수 있음

=> 제가 굉장히 많이 씁니다!!



"""
"""필요 라이브러리들 호출"""



import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd 

import seaborn as sns #시각화를 위한 라이브러리

import matplotlib.pyplot as plt

import calendar 

from datetime import datetime



import os

print(os.listdir("../input"))
"""

1) 훈련, 테스트 데이터셋의 개괄적인 형태 및 데이터의 컬럼의 속성 및 값의 개수 파악

"""



#훈련데이터와 테스트 데이터 세트를 불러온다

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#훈련데이터 셋의 개괄적인 모형 파악

train.head()
#데이터 셋 내에 있는 컬럼 속성들에 대한 설명



"""

datetime - hourly date + timestamp  

season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 

holiday - whether the day is considered a holiday

workingday - whether the day is neither a weekend nor holiday

weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 

2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 

3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 

4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 

temp - temperature in Celsius

atemp - "feels like" temperature in Celsius

humidity - relative humidity

windspeed - wind speed

casual - number of non-registered user rentals initiated

registered - number of registered user rentals initiated

count - number of total rentals

"""



#훈련 데이터셋의 각 컬럼별 데이터타입 및 값의 갯수 파악

train.info()
#테스트 데이터 셋의 개괄적인 형태 출력

test.head()
""" 2) 데이터 전처리 및 시각화 """



#datetime속성을 분리하여 추출속성으로 활용하기 위해 split함수를 사용하여 년-월-일 과 시간을 분리한다.

train['tempDate'] = train.datetime.apply(lambda x:x.split())
#분리한 tempDate를 가지고 년-월-일을 이용하여 year,month,day 그리고 weekday column을 추출한다.

# split() 내장함수 설명: https://wikidocs.net/13 [문자형 자료형_ 문자열 나누기] <=> join() [문자형 자료형_ 문자열 삽입]

train['year'] = train.tempDate.apply(lambda x:x[0].split('-')[0])

train['month'] = train.tempDate.apply(lambda x:x[0].split('-')[1])

train['day'] = train.tempDate.apply(lambda x:x[0].split('-')[2])

#weekday는 calendar패키지와 datetime패키지를 활용한다.

#calendar.day_name 사용법 : https://stackoverflow.com/questions/36341484/get-day-name-from-weekday-int

#datetime.strptime 문서: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

#파이썬에서 날짜와 시간 다루기: https://datascienceschool.net/view-notebook/465066ac92ef4da3b0aba32f76d9750a/ 

train['weekday'] = train.tempDate.apply(lambda x:calendar.day_name[datetime.strptime(x[0],"%Y-%m-%d").weekday()])



train['hour'] = train.tempDate.apply(lambda x:x[1].split(':')[0])
#분리를 통해 추출된 속성은 문자열 속성을 가지고 있음 따라서 숫자형 데이터로 변환해 줄 필요가 있음.

#pandas.to_numeric(): https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_numeric.html

train['year'] = pd.to_numeric(train.year,errors='coerce')

train['month'] = pd.to_numeric(train.month,errors='coerce')

train['day'] = pd.to_numeric(train.day,errors='coerce')

train['hour'] = pd.to_numeric(train.hour,errors='coerce')
#year,month,day,hour가 숫자형으로 변환되었음을 알 수 있음.

train.info()
#필요를 다한 tempDate column을 drop함

train = train.drop('tempDate',axis=1)
#각각의 속성과 예측의 결과값으로 쓰이는 count값과의 관계 파악



#년도와 count

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,2,1)

ax1 = sns.barplot(x='year',y='count',data=train.groupby('year')['count'].mean().reset_index())



#month와 count

ax2 = fig.add_subplot(2,2,2)

ax2 = sns.barplot(x='month',y='count',data=train.groupby('month')['count'].mean().reset_index())



#day와 count

ax3 = fig.add_subplot(2,2,3)

ax3 = sns.barplot(x='day',y='count',data=train.groupby('day')['count'].mean().reset_index())



#hour와 count

ax4 = fig.add_subplot(2,2,4)

ax4 = sns.barplot(x='hour',y='count',data=train.groupby('hour')['count'].mean().reset_index())
#계절과 count

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,2,1)

ax1 = sns.barplot(x='season',y='count',data=train.groupby('season')['count'].mean().reset_index())



#휴일 여부와 count

ax2 = fig.add_subplot(2,2,2)

ax2 = sns.barplot(x='holiday',y='count',data=train.groupby('holiday')['count'].mean().reset_index())



#작업일 여부와 count

ax3 = fig.add_subplot(2,2,3)

ax3 = sns.barplot(x='workingday',y='count',data=train.groupby('workingday')['count'].mean().reset_index())



#날씨와 count

ax4 = fig.add_subplot(2,2,4)

ax4 = sns.barplot(x='weather',y='count',data=train.groupby('weather')['count'].mean().reset_index())
"""

해당 부분은 필자가 스스로 데이터를 보고 이상함을 느껴 전처리함.

왜냐하면, 처음 import한 데이터 셋에서 head()를 하였을 때 1월1일의 season column은 1 즉 봄을 가르키는데,

직접 3월에 washington을 직접 가본 결과 1월은 확실히 겨울이다.

따라서 아래의 badToRight를 이용하여 season column을 수정하고자 했음.

이 데이터 때문에 참조했던 커널과는 다른 정확도를 나타낼 수 있음.

"""



def badToRight(month):

    if month in [12,1,2]:

        return 4

    elif month in [3,4,5]:

        return 1

    elif month in [6,7,8]:

        return 2

    elif month in [9,10,11]:

        return 3



#apply() 내장함수는 split(),map(),join(),filter()등 과 함꼐 필수적으로 숙지해야 할 함수이다.

train['season'] = train.month.apply(badToRight)
#위의 시각화와 같이 하나의 컬럼과 결과 값을 비교해보자



#계절과 count

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,2,1)

ax1 = sns.barplot(x='season',y='count',data=train.groupby('season')['count'].mean().reset_index())



#휴일 여부와 count

ax2 = fig.add_subplot(2,2,2)

ax2 = sns.barplot(x='holiday',y='count',data=train.groupby('holiday')['count'].mean().reset_index())



#작업일 여부와 count

ax3 = fig.add_subplot(2,2,3)

ax3 = sns.barplot(x='workingday',y='count',data=train.groupby('workingday')['count'].mean().reset_index())



#날씨와 count

ax4 = fig.add_subplot(2,2,4)

ax4 = sns.barplot(x='weather',y='count',data=train.groupby('weather')['count'].mean().reset_index())
#그리고 남은 분포를 통해 표현하였을 때 좋은 컬럼들을 count와 비교해보자



#온도와 count

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,2,1)

ax1 = sns.distplot(train.temp,bins=range(train.temp.min().astype('int'),train.temp.max().astype('int')+1))



#평균온도와 count

ax2 = fig.add_subplot(2,2,2)

ax2 = sns.distplot(train.atemp,bins=range(train.atemp.min().astype('int'),train.atemp.max().astype('int')+1))



#습도와 count

ax3 = fig.add_subplot(2,2,3)

ax3 = sns.distplot(train.humidity,bins=range(train.humidity.min().astype('int'),train.humidity.max().astype('int')+1))



#바람속도와 count

ax4 = fig.add_subplot(2,2,4)

ax4 = sns.distplot(train.windspeed,bins=range(train.windspeed.min().astype('int'),train.windspeed.max().astype('int')+1))
#각각의 컬럼들 간의 상관계수를 heatmap을 통해 시각화



fig = plt.figure(figsize=[20,20])

ax = sns.heatmap(train.corr(),annot=True,square=True)
#heatmap 상관관계를 참조하여 이전의 시각화와는 달리 두 개의 서로다른 컬럼이 적용된 count를 시각화해보자



#시간과 계절에 따른 count

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,2,1)

ax1 = sns.pointplot(x='hour',y='count',hue='season',data=train.groupby(['season','hour'])['count'].mean().reset_index())



#시간과 휴일 여부에 따른 count

ax2 = fig.add_subplot(2,2,2)

ax2 = sns.pointplot(x='hour',y='count',hue='holiday',data=train.groupby(['holiday','hour'])['count'].mean().reset_index())



#시간과 휴일 여부에 따른 count

ax3 = fig.add_subplot(2,2,3)

ax3 = sns.pointplot(x='hour',y='count',hue='weekday',hue_order=['Sunday','Monday','Tuesday','Wendnesday','Thursday','Friday','Saturday'],data=train.groupby(['weekday','hour'])['count'].mean().reset_index())



#시간과 날씨에 따른 count

ax4 = fig.add_subplot(2,2,4)

ax4 = sns.pointplot(x='hour',y='count',hue='weather',data=train.groupby(['weather','hour'])['count'].mean().reset_index())
#마지막 시각화에 이상치가 있는 것같아서 확인



train[train.weather==4]
#달과 날씨에 따른 count 

fig = plt.figure(figsize=[12,10])

ax1 = fig.add_subplot(2,1,1)

ax1 = sns.pointplot(x='month',y='count',hue='weather',data=train.groupby(['weather','month'])['count'].mean().reset_index())



#달별 count

ax2 = fig.add_subplot(2,1,2)

ax2 = sns.barplot(x='month',y='count',data=train.groupby('month')['count'].mean().reset_index())
"""

Windspeed 분포를 표현한 그래프에서 Windspeed가 0인 값들이 많았는데,

이는 실제로 0이었던지 or 값을 제대로 측정하지 못해서 0인지 두 개의 경우가 있다.

하지만 후자의 생각을 가지고 우리의 데이터를 활용하여 windspeed값을 부여해보자

"""



#머신러닝 모델에 훈련시킬 때는 문자열 값은 불가능하기 때문에 문자열을 카테고리화 하고 각각에 해당하는 값을 숫자로 변환해준다

train['weekday']= train.weekday.astype('category')
print(train['weekday'].cat.categories)
#0:Sunday --> 6:Saturday

train.weekday.cat.categories = ['5','1','6','0','4','2','3']
"""

RandomForest를 활용하여 Windspeed값을 부여해보자

하나의 데이터를 Windspeed가 0인 그리고 0이 아닌 데이터프레임으로 분리하고

학습시킬 0이 아닌 데이터 프레임에서는 Windspeed만 담긴 Series와 이외의 학습시킬 column들의 데이터프레임으로 분리한다

학습 시킨 후에 Windspeed가 0인 데이터 프레임에서 학습시킨 컬럼과 같게 추출하여 결과 값을 부여받은 후,

Windspeed가 0인 데이터프레임에 Windspeed값을 부여한다.

"""

from sklearn.ensemble import RandomForestRegressor



#Windspeed가 0인 데이터프레임

windspeed_0 = train[train.windspeed == 0]

#Windspeed가 0이 아닌 데이터프레임

windspeed_Not0 = train[train.windspeed != 0]



#Windspeed가 0인 데이터 프레임에 투입을 원치 않는 컬럼을 배제

windspeed_0_df = windspeed_0.drop(['windspeed','casual','registered','count','datetime'],axis=1)



#Windspeed가 0이 아닌 데이터 프레임은 위와 동일한 데이터프레임을 형성하고 학습시킬 Windspeed Series를 그대로 둠

windspeed_Not0_df = windspeed_Not0.drop(['windspeed','casual','registered','count','datetime'],axis=1)

windspeed_Not0_series = windspeed_Not0['windspeed'] 



#모델에 0이 아닌 데이터프레임과 결과값을 학습

rf = RandomForestRegressor()

rf.fit(windspeed_Not0_df,windspeed_Not0_series)

#학습된 모델에 Windspeed가 0인 데이터프레임의 Windspeed를 도출

predicted_windspeed_0 = rf.predict(windspeed_0_df)

#도출된 값을 원래의 데이터프레임에 삽입

windspeed_0['windspeed'] = predicted_windspeed_0
#나눈 데이터 프레임을 원래의 형태로 복원

train = pd.concat([windspeed_0,windspeed_Not0],axis=0)
#시간별 정렬을 위해 string type의 datetime을 datetime으로 변환

train.datetime = pd.to_datetime(train.datetime,errors='coerce')
#합쳐진 데이터를 datetime순으로 정렬

train = train.sort_values(by=['datetime'])
#windspeed를 수정한 후 다시 상관계수를 분석

#우리의 기대와는 달리 windspeed와 count의 상관관계는 0.1에서 0.11로 간소한 차이만 보임.

fig = plt.figure(figsize=[20,20])

ax = sns.heatmap(train.corr(),annot=True,square=True)
fig = plt.figure(figsize=[5,5])

sns.distplot(train['windspeed'],bins=np.linspace(train['windspeed'].min(),train['windspeed'].max(),10))

plt.suptitle("Filled by Random Forest Regressor")

print("Min value of windspeed is {}".format(train['windspeed'].min()))
"""이제 모든 동일한 전처리 과정을 test셋과 한꺼번에 진행"""

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
combine = pd.concat([train,test],axis=0)
combine.info()
combine['tempDate'] = combine.datetime.apply(lambda x:x.split())

combine['weekday'] = combine.tempDate.apply(lambda x: calendar.day_name[datetime.strptime(x[0],"%Y-%m-%d").weekday()])

combine['year'] = combine.tempDate.apply(lambda x: x[0].split('-')[0])

combine['month'] = combine.tempDate.apply(lambda x: x[0].split('-')[1])

combine['day'] = combine.tempDate.apply(lambda x: x[0].split('-')[2])

combine['hour'] = combine.tempDate.apply(lambda x: x[1].split(':')[0])
combine['year'] = pd.to_numeric(combine.year,errors='coerce')

combine['month'] = pd.to_numeric(combine.month,errors='coerce')

combine['day'] = pd.to_numeric(combine.day,errors='coerce')

combine['hour'] = pd.to_numeric(combine.hour,errors='coerce')
combine.info()
combine['season'] = combine.month.apply(badToRight)
combine.head()
combine.weekday = combine.weekday.astype('category')
combine.weekday.cat.categories = ['5','1','6','0','4','2','3']
dataWind0 = combine[combine['windspeed']==0]

dataWindNot0 = combine[combine['windspeed']!=0]
dataWind0.columns
dataWind0_df = dataWind0.drop(['windspeed','casual','registered','count','datetime','tempDate'],axis=1)



dataWindNot0_df = dataWindNot0.drop(['windspeed','casual','registered','count','datetime','tempDate'],axis=1)

dataWindNot0_series = dataWindNot0['windspeed']
dataWindNot0_df.head()
dataWind0_df.head()
rf2 = RandomForestRegressor()

rf2.fit(dataWindNot0_df,dataWindNot0_series)

predicted = rf2.predict(dataWind0_df)

print(predicted)
dataWind0['windspeed'] = predicted
combine = pd.concat([dataWind0,dataWindNot0],axis=0)
#우리가 가진 column들 중 값들이 일정하고 정해져있다면 category로 변경해주고

#필요하지 않은 column들은 이제 버린다.

categorizational_columns = ['holiday','humidity','season','weather','workingday','year','month','day','hour']

drop_columns = ['datetime','casual','registered','count','tempDate']
#categorical하게 변환

for col in categorizational_columns:

    combine[col] = combine[col].astype('category')
#합쳐진 combine데이터 셋에서 count의 유무로 훈련과 테스트셋을 분리하고 각각을 datetime으로 정렬

train = combine[pd.notnull(combine['count'])].sort_values(by='datetime')

test = combine[~pd.notnull(combine['count'])].sort_values(by='datetime')



#데이터 훈련시 집어 넣게 될 각각의 결과 값들

datetimecol = test['datetime']

yLabels = train['count'] #count

yLabelsRegistered = train['registered'] #등록된 사용자

yLabelsCasual = train['casual'] #임시 사용자
#필요 없는 column들을 버린 후의 훈련과 테스트 셋

train = train.drop(drop_columns,axis=1)

test = test.drop(drop_columns,axis=1)
"""

해당 문제에서는 RMSLE방식을 이용하여 제대로 예측이 되었는지 평가하게 됨.

RMSLE는 아래 링크를 참조하여 이용.

https://programmers.co.kr/learn/courses/21/lessons/943#



RMSLE

과대평가 된 항목보다는 과소평가 된 항목에 페널티를 주는방식

오차를 제곱하여 형균한 값의 제곱근으로 값이 작아질 수록 정밀도가 높음

0에 가까운 값이 나올 수록 정밀도가 높다

"""



# y is predict value y_ is actual value

def rmsle(y, y_,convertExp=True):

    if convertExp:

        y = np.exp(y), 

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
#선형 회귀 모델

#선형 회귀모델은 건드릴 만한 내부 attr들이 없음

from sklearn.linear_model import LinearRegression,Ridge,Lasso





lr = LinearRegression()



"""

아래의 커널을 참조하여 yLabels를 로그화 하려는데 왜 np.log가 아닌 np.log1p를 활용하는가??

np.log1p는 np.log(1+x)와 동일. 이유는 만약 어떤 x값이 0인데 이를 log하게되면, (-)무한대로 수렴하기 때문에 np.log1p를 활용함. 

참조: https://ko.wikipedia.org/wiki/%EB%A1%9C%EA%B7%B8 

"""

yLabelslog = np.log1p(yLabels)

#선형 모델에 우리의 데이터를 학습

lr.fit(train,yLabelslog)

#결과 값 도출

preds = lr.predict(train)

#rmsle함수의 element에 np.exp()지수 함수를 취하는 이유는 우리의 preds값에 얻어진 것은 한번 log를 한 값이기 때문에 원래 모델에는 log를 하지 않은 원래의 값을 넣기 위함임.

print('RMSLE Value For Linear Regression: {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
"""

데이터 훈련시 Log값을 취하는 이유??

우리가 결과 값으로 투입하는 Count값이 최저 값과 최고 값의 낙폭이 너무 커서

만약 log를 취하지 않고 해보면 print하는 결과 값이 inf(infinity)로 뜨게 됨

"""



#count값의 분포

sns.distplot(yLabels,bins=range(yLabels.min().astype('int'),yLabels.max().astype('int')))



#기존 훈련 데이터셋의 count의 개수

print(yLabels.count()) #10886



""" 

3 sigma를 활용한 이상치 확인

참조 : https://ko.wikipedia.org/wiki/68-95-99.7_%EA%B7%9C%EC%B9%99

"""

#3시그마를 적용한 이상치를 배제한 훈련 데이터셋의 count의 개수

yLabels[np.logical_and(yLabels.mean()-3*yLabels.std() <= yLabels,yLabels.mean()+3*yLabels.std() >= yLabels)].count() #10739

#이상치들이 존재할 때는 log를 활용하여 값을 도출
"""

GridSearchCV를 활용하면 우리가 이용하게 될 각각의 모델마다 변경해야 하는 파라미터 튜닝시 어떤 파라미터가 최적의 값을 내는지 등을 알 수 있음.



GridSearchCV 참조:

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

https://datascienceschool.net/view-notebook/ff4b5d491cc34f94aea04baca86fbef8/

"""

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



#Ridge모델은 L2제약을 가지는 선형회귀모델에서 개선된 모델이며 해당 모델에서 유의 깊게 튜닝해야하는 파라미터는 alpha값이다.

ridge = Ridge()



#우리가 튜닝하고자하는 Ridge의 파라미터 중 특정 파라미터에 배열 값으로 넘겨주게 되면 테스트 후 어떤 파라미터가 최적의 값인지 알려줌 

ridge_params = {'max_iter':[3000],'alpha':[0.001,0.01,0.1,1,10,100,1000]}

rmsle_scorer = metrics.make_scorer(rmsle,greater_is_better=False)

grid_ridge = GridSearchCV(ridge,ridge_params,scoring=rmsle_scorer,cv=5)



grid_ridge.fit(train,yLabelslog)

preds = grid_ridge.predict(train)

print(grid_ridge.best_params_)

print('RMSLE Value for Ridge Regression {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
#결과에 대해 GridSearchCV의 변수인 grid_ridge변수에 cv_result_를 통해 alpha값의 변화에 따라 평균값의 변화를 파악 가능

df = pd.DataFrame(grid_ridge.cv_results_)
df.head()
#Ridge모델은 L1제약을 가지는 선형회귀모델에서 개선된 모델이며 해당 모델에서 유의 깊게 튜닝해야하는 파라미터는 alpha값이다.

lasso = Lasso()



lasso_params = {'max_iter':[3000],'alpha':[0.001,0.01,0.1,1,10,100,1000]}

grid_lasso = GridSearchCV(lasso,lasso_params,scoring=rmsle_scorer,cv=5)

grid_lasso.fit(train,yLabelslog)

preds = grid_lasso.predict(train)

print('RMSLE Value for Lasso Regression {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
rf = RandomForestRegressor()



rf_params = {'n_estimators':[1,10,100]}

grid_rf = GridSearchCV(rf,rf_params,scoring=rmsle_scorer,cv=5)

grid_rf.fit(train,yLabelslog)

preds = grid_rf.predict(train)

print('RMSLE Value for RandomForest {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
from sklearn.ensemble import GradientBoostingRegressor



gb = GradientBoostingRegressor()

gb_params={'max_depth':range(1,11,1),'n_estimators':[1,10,100]}

grid_gb=GridSearchCV(gb,gb_params,scoring=rmsle_scorer,cv=5)

grid_gb.fit(train,yLabelslog)

preds = grid_gb.predict(train)

print('RMSLE Value for GradientBoosting {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
predsTest = grid_gb.predict(test)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sns.distplot(yLabels,ax=ax1,bins=50)

sns.distplot(np.exp(predsTest),ax=ax2,bins=50)
submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(predsTest)]

    })

submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)