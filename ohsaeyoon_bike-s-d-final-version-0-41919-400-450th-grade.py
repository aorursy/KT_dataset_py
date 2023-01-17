import pandas as pd

import numpy as np

%matplotlib inline 

import matplotlib.pyplot as plt

import seaborn as sns

pd.options.display.max_columns = 50

sns.set_style("whitegrid")



import matplotlib

matplotlib.rc("font", family = "AppleGothic")

matplotlib.rc("axes", unicode_minus = False)



from IPython.display import set_matplotlib_formats

set_matplotlib_formats("retina")
train=pd.read_csv("../input/train.csv",parse_dates=["datetime"])

print(train.shape)

train.head(2)
test=pd.read_csv("../input/test.csv",parse_dates=["datetime"])

print(test.shape)

test.head(2)
# datetime을 정확히 알기위한 사전작업 실시



train["datetime"]=pd.to_datetime(train["datetime"])

train["datetime_year"]=train["datetime"].dt.year

train["datetime_month"]=train["datetime"].dt.month

train["datetime_day"]=train["datetime"].dt.day

train["datetime_hour"]=train["datetime"].dt.hour

train["datetime_minute"]=train["datetime"].dt.minute

train["datetime_second"]=train["datetime"].dt.second

train["datetime_dayofweek"]=train["datetime"].dt.dayofweek



train.head(2)

test["datetime"]=pd.to_datetime(test["datetime"])



test["datetime_year"]=test["datetime"].dt.year

test["datetime_month"]=test["datetime"].dt.month

test["datetime_day"]=test["datetime"].dt.day

test["datetime_hour"]=test["datetime"].dt.hour

test["datetime_minute"]=test["datetime"].dt.minute

test["datetime_second"]=test["datetime"].dt.second

test["datetime_dayofweek"]=test["datetime"].dt.dayofweek
# train의 windspeed부분에서 0부분이 지나치게 높다

## 이를 통해 이 windspeed 부분에 문제가 있으며 / 이를 기존 주어진 정보를 통해 0에 있는걸 파악하는 작업이 선행되야 함을 의미

sns.distplot(train["windspeed"])
# train에서 windspeed의 수가 



windspeed_0=train.loc[train["windspeed"]==0]

windspeed_1=train.loc[train["windspeed"]!=0]

print(windspeed_0.shape)

print(windspeed_1.shape)

print(train.shape)

train.head(2)
feature=["season","holiday","workingday","weather","temp","atemp","humidity","datetime_month","datetime_year",

         "datetime_day","datetime_hour","datetime_dayofweek"]

label=["windspeed"]



x_windspeed_1=windspeed_1[feature]

y_windspeed_1=windspeed_1[label]

x_windspeed_0=windspeed_0[feature]

x_windspeed_0.head()
from sklearn.ensemble import RandomForestRegressor

s_model=RandomForestRegressor(n_jobs=-1,random_state=37)

s_model.fit(x_windspeed_1,y_windspeed_1)



s_prediction=s_model.predict(x_windspeed_0)

windspeed_0["windspeed"]=s_prediction

windspeed_0.head()

train=pd.concat([windspeed_0,windspeed_1],axis=0)

train.loc[train["windspeed"]==0]
# 각 상관관계를 통해 어떤 것이 연관성이 높은지 조사해본다

# hour의 상관성이 제일 높다. 이를 분석해볼 필요가 있다 



fig=plt.figure(figsize=[20,20])

sns.heatmap(train.corr(),annot=True,square=True)
fig=plt.figure(figsize=[10,10])

ax1=fig.add_subplot(2,2,1)

ax1=sns.pointplot(x="datetime_hour",y="count",hue="datetime_year",data=train.groupby(["datetime_hour","datetime_year"])["count"].mean().reset_index())

# 2011,2012 둘다 동일한 흐름을 보인다



ax1=fig.add_subplot(2,2,2)

ax1=sns.pointplot(x="datetime_hour",y="count",hue="workingday",data=train.groupby(["datetime_hour","workingday"])["count"].mean().reset_index())

# 일하는날 자전거 타는 비중이 절대적으로 높다 / 10~16시까지는 일안하는날 타는 비중이 더 높다 



ax1=fig.add_subplot(2,2,3)

ax1=sns.pointplot(x="datetime_hour",y="count",hue="holiday",data=train.groupby(["datetime_hour","holiday"])["count"].mean().reset_index())

# 휴일 역시 10-16시 비중이 높음. 이는 주말과 휴일이 겹친날이 어느정도 많이 있고 -> 이에 따른 결과일 가능성이 높음을 나타낸다



ax1=fig.add_subplot(2,2,4)

ax1=sns.pointplot(x="datetime_hour",y="count",hue="season",data=train.groupby(["datetime_hour","season"])["count"].mean().reset_index())
# 따로 하면 나오는게 없고 특히 2012년에 count절대수가 더 많음에도 불구하고 총 그래프는 7,8,9에 특히 2011년걸 따라가는게 수상하다

# 이럴 경우에는 이 두개를 합쳐서 추이를 따져봐야한다는말임
fig=plt.figure(figsize=[18,12])

ax1=fig.add_subplot(2,2,1)

ax1=sns.pointplot(x="datetime_month",y="count",data=train.groupby(["datetime_month"])["count"].sum().reset_index())



ax1=fig.add_subplot(2,2,2)

ax1=sns.pointplot(x="datetime_month",y="count",hue="datetime_year",data=train.groupby(["datetime_month","datetime_year"])["count"].sum().reset_index())

# 지금 이 그래프로는 영향 및 의문점을 찾을 수가 없다. 

# 2011년보다 2012년이 모두 더 높다는점 / 마찬가지로 시즌으로 나누어봐도 2012년이 더 높다

# 이는 단순하게 년도나 시즌으로 나누어서 보는게 아니라 합쳐서 시간적인 부분을 파악해야 한다는 것이다



fig=plt.figure(figsize=[12,12])

ax1=fig.add_subplot(2,2,1)

ax1=sns.barplot(x="season",y="count",hue="datetime_year",\

                data=train.groupby(["season","datetime_year"])["count"].sum().reset_index())



ax1=fig.add_subplot(2,2,2)

ax1=sns.barplot(x="weather",y="count",hue="datetime_year",\

                data=train.groupby(["weather","datetime_year"])["count"].sum().reset_index())
train["datetime_year(str)"]=train["datetime_year"].astype("str")

train["datetime_month(str)"]=train["datetime_month"].astype("str")

train["datetime_year_month"]=train["datetime_year"].astype("str")+"-"+train["datetime_month"].astype("str")

train.head()
# 달을 합쳐서 시간순으로 보았음.



fig=plt.figure(figsize=[20,5])

ax1=fig.add_subplot(1,1,1)

ax1=sns.barplot(x="datetime_year_month",y="count",data=train)



# 2011년에는 7-9월의 하락이 있었느나 2012년은 성장 -> 잘했다는 것
fig=plt.figure(figsize=[15,5])

ax1=fig.add_subplot(1,1,1)

ax1=sns.pointplot(x="datetime_hour",y="count",hue="datetime_dayofweek",data=train.groupby(["datetime_hour","datetime_dayofweek"])["count"].mean().reset_index())



# 월요일 금요일중 누가 더 휴무를 많이내고 누가더 자전거 많이타냐



# 0-4hour -> 토요일새벽부터 일요일새벽까지 타는 인원이 제일 많음 / 금요일새벽-토요일아침 / 목요일새벽-금요일새벽

# 4-5hour -> 평일 대부분 일한다 

# 5hour -> 5시 이후부터 대부분 출근이 시작됨 그래서 자전거 수요가 급증 / 월금은 연차나 휴가낸 사람들이 많다는걸 반증

# 6-8시 본격적인 출근시간인데 -> 수목이 많다는건 월화는 대중교통을 이용할 수 있다는 이야기 

# 금토일로 연차나 휴가를 낸 사람들이 토일월로 낸 사람보다 자전거 더 많이 이용 -> 운동을 더 많이 할수도 있고 더 다른데 갈수도 잇고 

# 그래프상 화요일이 연치가 가장 많을걸로 추측할 수 있다 ->holiday그래프와 유사함

# 그래프상에서보면 토일월<금토일로 연차를 더 많이 낸 사람들이 자전거를 많이타고 /월요일은 새벽에는 힘들어서 그냥 대중교통 후 퇴근부터 자전거이용

# 금요일날 퇴근하고 집가지 않고 놀거나 / 연차낸사람들이 오후에 타고 지금은 안타거나 

week = pd.pivot_table(index=["datetime_dayofweek"],values="count",data=train,aggfunc=np.sum)



# 주말을 들여다보았을 때 토요일이 가장 높고 일요일이 제일 낮다 dayofweek는 매우 중요한 포인트라 할수 있음

week.plot(kind = "line", color = "skyblue")
fig=plt.figure(figsize=[15,10])

ax1=fig.add_subplot(2,2,1)

ax1=sns.distplot(train["temp"])



ax1=fig.add_subplot(2,2,2)

ax1=sns.distplot(train["atemp"])



ax1=fig.add_subplot(2,2,3)

ax1=sns.distplot(train["humidity"])



ax1=fig.add_subplot(2,2,4)

ax1=sns.distplot(train["windspeed"])
# 공식을 찾아보고 이를 대입시켜서 

## 바람의 강도세기에 따른 이상적 수치를 찾아본다 



windchill=0.6215*(train["temp"])+35.74-35.75*(train["windspeed"]**0.16)+0.4275*(train["windspeed"]**0.16)

windchill[0:10]
train["windchill"]=windchill

train.head()
windchill=0.6215*(test["temp"])+35.74-35.75*(test["windspeed"]**0.16)+0.4275*(test["windspeed"]**0.16)

test["windchill"]=windchill

test.head()
# 기존 데이터를 활용하여 새로운 정보를 추출해낸다

##  temp / humidity를 통해 새로운 공식을 대입하여 새로운 조건을 생성해본다



train["bad_humidity"]=9/5*train["temp"]-0.55*(1-train["humidity"]/100)*(9/5*train["temp"]-26)+32

test["bad_humidity"]=9/5*test["temp"]-0.55*(1-test["humidity"]/100)*(9/5*test["temp"]-26)+32



train.head()

test.head()
train.loc[train["bad_humidity"]>80,"humidity2"]="Terrible"

train.loc[train["bad_humidity"]<80,"humidity2"]="Bad"

train.loc[train["bad_humidity"]<75,"humidity2"]="SOSO"

train.loc[train["bad_humidity"]<68,"humidity2"]="Good"

train.head()
test.loc[test["bad_humidity"]>80,"humidity2"]="Terrible"

test.loc[test["bad_humidity"]<80,"humidity2"]="Bad"

test.loc[test["bad_humidity"]<75,"humidity2"]="SOSO"

test.loc[test["bad_humidity"]<68,"humidity2"]="Good"

test.head()
# 그래프를 통해서 

## 계산된 날씨가 좋은 시기에는 자전거 이용수가 많고 / 좋지 않은 날에는 이용률이 낮아진다 

## 첫번째 그래프를 통해 good을 feature_name의 변수로 적용할 수 있으나 / 이외 나머지는 넣어도 크게 변화를 줄 수 있는 여부가 없다

## soso의 경우 good과의 갭차이가 다른 두 변수의 갭차이보다 훨씬 높기에 크게 결정적인 변수라고 보기 어렵다



### 두번째 그래프에서 중요한 것은 terrible인데 날씨가 좋지 못한 여름임에도 불구하고 terrible인 시기에 이용수가 매우 높다

### 적어도 7,8월달에 한해서는 날씨의 중요성이 떨어지고 / 



### 오히려 지나치게 더운 날씨로 인해 자전거를 이용하여 출퇴근 하는 사람들이 있을 수 있고 / 방학기간 학생들의 이용수가 급등할 수도 있다

### 날씨만 보면 terrible비중이 높은 여름의 이용수가 적다고 생각할 수 있지만 / 이 그래프를 통해

### 오히려 여름때 이들을 타겟으로 할 수 있는 여러가지 프로모션을 개발한다면 이용자 수를 급증 시킬 수 있음을 볼 수 있다 





fig=plt.figure(figsize=[10,10])

ax1=fig.add_subplot(2,2,1)

ax1=sns.barplot(x="humidity2",y="count",data=train.groupby(["humidity2"])["count"].sum().reset_index())



ax1=fig.add_subplot(2,2,2)

ax1=sns.pointplot(x="datetime_month",y="count",hue="humidity2",\

                  data=train.groupby(["datetime_month","humidity2"])["count"].sum().reset_index())
feature_names=["season","holiday","workingday","weather",

               "temp","humidity","windspeed","datetime_year",

               "datetime_hour","datetime_dayofweek","atemp"]

label_names="count"

x_train=train[feature_names]

y_train=train[label_names]

x_test=test[feature_names]
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(n_jobs=-1,random_state=37,n_estimators=100)

model
from sklearn.metrics import make_scorer

def rmle(predict,actual):

    predict=np.array(predict)

    actual=np.array(actual)

    log_predict=np.log(predict+1)

    log_actual=np.log(actual+1)

    distance=(log_predict-log_actual)**2

    mean_distance=distance.mean()

    score=np.sqrt(mean_distance)

    return score



rmsle=make_scorer(rmle)

rmsle
from sklearn.model_selection import cross_val_score

score=cross_val_score(model,x_train,y_train,cv=20,scoring=rmsle).mean()

score