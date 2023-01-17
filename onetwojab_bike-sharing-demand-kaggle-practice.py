import pandas as pd

import numpy as np



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



# 노트북 안에 그래프를 그리기 위해

%matplotlib inline



# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처

mpl.rcParams['axes.unicode_minus']=False



import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv("../input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

train.shape

test=pd.read_csv("../input/bike-sharing-demand/test.csv", parse_dates=["datetime"])

test.shape
train["year"]=train["datetime"].dt.year

train["month"]=train["datetime"].dt.month

train["day"]=train["datetime"].dt.day

train["hour"]=train["datetime"].dt.hour

train["minute"]=train["datetime"].dt.minute

train["second"]=train["datetime"].dt.second

train["dayofweek"]=train["datetime"].dt.dayofweek

train.shape
test["year"]=test["datetime"].dt.year

test["month"]=test["datetime"].dt.month

test["day"]=test["datetime"].dt.day

test["hour"]=test["datetime"].dt.hour

test["minute"]=test["datetime"].dt.minute

test["second"]=test["datetime"].dt.second

test["dayofweek"]=test["datetime"].dt.dayofweek

test.shape
# widspeed 풍속에 0 값이 가장 많다. => 잘못 기록된 데이터를 고쳐 줄 필요가 있음

fig, axes = plt.subplots(nrows=2)

fig.set_size_inches(18,10)



plt.sca(axes[0])

plt.xticks(rotation=30, ha='right')

axes[0].set(ylabel='Count',title="train windspeed")

sns.countplot(data=train, x="windspeed", ax=axes[0])



plt.sca(axes[1])

plt.xticks(rotation=30, ha='right')

axes[1].set(ylabel='Count',title="test windspeed")

sns.countplot(data=test, x="windspeed", ax=axes[1])
# 풍속의 0값에 특정 값을 넣어준다.

# 평균을 구해 일괄적으로 넣어줄 수도 있지만, 예측의 정확도를 높이는 데 도움이 될것 같진 않다.

# train.loc[train["windspeed"] == 0, "windspeed"] = train["windspeed"].mean()

# test.loc[train["windspeed"] == 0, "windspeed"] = train["windspeed"].mean()
# 풍속이 0인것과 아닌 것의 세트를 나누어 준다.

trainWind0 = train.loc[train['windspeed'] == 0]

trainWindNot0 = train.loc[train['windspeed'] != 0]

print(trainWind0.shape)

print(trainWindNot0.shape)
# 그래서 머신러닝으로 예측을 해서 풍속을 넣어주도록 한다.

from sklearn.ensemble import RandomForestClassifier



def predict_windspeed(data):

    

    # 풍속이 0인것과 아닌 것을 나누어 준다.

    dataWind0 = data.loc[data['windspeed'] == 0]

    dataWindNot0 = data.loc[data['windspeed'] != 0]

    

    # 풍속을 예측할 피처를 선택한다.

    wCol = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]



    # 풍속이 0이 아닌 데이터들의 타입을 스트링으로 바꿔준다.

    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")



    # 랜덤포레스트 분류기를 사용한다.

    rfModel_wind = RandomForestClassifier()



    # wCol에 있는 피처의 값을 바탕으로 풍속을 학습시킨다.

    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])



    # 학습한 값을 바탕으로 풍속이 0으로 기록 된 데이터의 풍속을 예측한다.

    wind0Values = rfModel_wind.predict(X = dataWind0[wCol])



    # 값을 다 예측 후 비교해 보기 위해

    # 예측한 값을 넣어 줄 데이터 프레임을 새로 만든다.

    predictWind0 = dataWind0

    predictWindNot0 = dataWindNot0



    # 값이 0으로 기록 된 풍속에 대해 예측한 값을 넣어준다.

    predictWind0["windspeed"] = wind0Values



    # dataWindNot0 0이 아닌 풍속이 있는 데이터프레임에 예측한 값이 있는 데이터프레임을 합쳐준다.

    data = predictWindNot0.append(predictWind0)



    # 풍속의 데이터타입을 float으로 지정해 준다.

    data["windspeed"] = data["windspeed"].astype("float")



    data.reset_index(inplace=True)

    data.drop('index', inplace=True, axis=1)

    

    return data
# 0값을 조정한다.

train = predict_windspeed(train)

# test = predict_windspeed(test)



# widspeed 의 0값을 조정한 데이터를 시각화

fig, ax1 = plt.subplots()

fig.set_size_inches(18,6)



plt.sca(ax1)

# 글씨 30도 기울기 시켜서 겹쳐보이지 않도록 해줌

plt.xticks(rotation=30, ha='right')

ax1.set(ylabel='Count',title="train windspeed")

sns.countplot(data=train, x="windspeed", ax=ax1)
# 연속형 feature와 범주형 feature 

# 연속형 feature = ["temp","humidity","windspeed","atemp"]

# 범주형 feature의 type을 category로 변경 해 준다. weather 1,2,3,4 봄 2배=> 가을 되는거 아님. 

# 범주형 feature는 one-hot-encodding 주로 많이 쓴다.

categorical_feature_names = ["season","holiday","workingday","weather",

                             "dayofweek","month","year","hour"]



for var in categorical_feature_names:

    train[var] = train[var].astype("category")

    test[var] = test[var].astype("category")
# feature 선택

feature_names = ["season", "weather", "temp", "atemp", "humidity", "windspeed",

                 "year", "hour", "dayofweek", "holiday", "workingday"]



feature_names
# 새로운 dateset으로 행렬 만들기

X_train = train[feature_names]



print(X_train.shape)

X_train.head()
X_test = test[feature_names]



print(X_test.shape)

X_test.head()
label_name = "count"



y_train = train[label_name]



print(y_train.shape)

y_train.head()
from sklearn.metrics import make_scorer



def rmsle(predicted_values, actual_values):

    # 넘파이로 배열 형태로 바꿔준다.

    predicted_values = np.array(predicted_values)

    actual_values = np.array(actual_values)

    

    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다.

    log_predict = np.log(predicted_values + 1)

    log_actual = np.log(actual_values + 1)

    

    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.

    difference = log_predict - log_actual

    # difference = (log_predict - log_actual) ** 2

    difference = np.square(difference)

    

    # 평균을 낸다.

    mean_difference = difference.mean()

    

    # 다시 루트를 씌운다.

    score = np.sqrt(mean_difference)

    

    return score



rmsle_scorer = make_scorer(rmsle)

rmsle_scorer
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.ensemble import RandomForestRegressor



max_depth_list = []



# n_estimators 값 높이면 시간 오래걸림;; 일단 100으로 초기화

model = RandomForestRegressor(n_estimators=100,

                              n_jobs=-1,

                              random_state=0)

model

%time score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)

score = score.mean()

# 0에 근접할수록 좋은 데이터

print("Score= {0:.5f}".format(score))
# 학습시킴, 피팅(옷을 맞출 때 사용하는 피팅을 생각함) - 피처와 레이블을 넣어주면 알아서 학습을 함

model.fit(X_train, y_train)
# 예측

predictions = model.predict(X_test)



print(predictions.shape)

predictions[0:10]
# 예측한 데이터를 시각화 해본다. 

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sns.distplot(y_train,ax=ax1,bins=50)

ax1.set(title="train")

sns.distplot(predictions,ax=ax2,bins=50)

ax2.set(title="test")
submission = pd.read_csv("../input/bike-sharing-demand/sampleSubmission.csv")

submission



submission["count"] = predictions



print(submission.shape)

submission.head()
submission.to_csv("Score_{0:.5f}_sampleSubmission.csv".format(score), index=False)