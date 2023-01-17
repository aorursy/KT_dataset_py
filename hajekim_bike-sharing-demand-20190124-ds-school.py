import pandas as pd
# train 데이터를 가져오자.
train = pd.read_csv("../input/train.csv")

# train의 행렬의 수를 보자.
print(train.shape)

# train의 상위 5열을 보자.
train.head()
# test 데이터를 불러오자
test = pd.read_csv("../input/test.csv")

# test의 행렬의 수를 보자.
print(test.shape)

# test의 상위 5열을 보자.
test.head()
# 연월일시분초로 나누도록 하자.
# train["datetime"].dt.year
# 이대로 바로 쓰면 에러가 난다.
# 왜냐하면 String으로 들어가 있기 때문에.
# Datetime 타입으로 변경해줘야 한다.

# datetime 타입으로 변경하고 저정하자.
train["datetime"] = pd.to_datetime(train["datetime"])
# 열을 추가 하기 전에 train 데이터의 크기를 보자.
print(train.shape)

# "연 월 일 시 분 초"로 쪼개서 저장하자.
train["datetime-year"] = train["datetime"].dt.year
train["datetime-month"] = train["datetime"].dt.month
train["datetime-day"] = train["datetime"].dt.day
train["datetime-hour"] = train["datetime"].dt.hour
train["datetime-minute"] = train["datetime"].dt.minute
train["datetime-second"] = train["datetime"].dt.second

# 20180124 추가
train["datetime-dayofweek"] = train["datetime"].dt.dayofweek

# 열을 추가 후에 train 데이터의 크기를 보자.
print(train.shape)


# 제대로 들어갔는지 데이터를 확인해보자.
train[["datetime", "datetime-year", "datetime-month", "datetime-day",
       "datetime-hour", "datetime-minute", "datetime-second", "datetime-dayofweek"]].head()
# test의 datetime의 Type을 String에서 datetime으로 변경하자.
test["datetime"] = pd.to_datetime(test["datetime"])
# 열을 추가 하기 전에 train 데이터의 크기를 보자.
print(test.shape)

# datetime을 연 월 일 시 분 초 로 각각 열을 추가해서 저장하자.
test["datetime-year"] = test["datetime"].dt.year
test["datetime-month"] = test["datetime"].dt.month
test["datetime-day"] = test["datetime"].dt.day
test["datetime-hour"] = test["datetime"].dt.hour
test["datetime-minute"] = test["datetime"].dt.minute
test["datetime-second"] = test["datetime"].dt.second

# 20180124 추가
test["datetime-dayofweek"] = test["datetime"].dt.dayofweek

# 열을 추가 후에 train 데이터의 크기를 보자.
print(test.shape)

# 제대로 들어갔는지 데이터를 확인해보자.
test[["datetime", "datetime-year", "datetime-month", "datetime-day",
      "datetime-hour", "datetime-minute", "datetime-second", "datetime-dayofweek"]].head()
import seaborn as sns

# 시각화를 위해서 한다.
%matplotlib inline
sns.barplot(data=train, x="weather", y="count")
sns.lmplot(data=train, x="temp", y="atemp")
sns.distplot(train["windspeed"])
sns.barplot(data=train, x="datetime-year", y="count")
sns.barplot(data=train, x="datetime-month", y="count")
sns.barplot(data=train, x="datetime-day", y="count")
sns.barplot(data=train, x="datetime-hour", y="count")
sns.barplot(data=train, x="datetime-minute", y="count")
sns.barplot(data=train, x="datetime-second", y="count")
# Integer를 String으로 데이터 타입을 변경한다.
train["datetime-year_month"] = train["datetime-year"].astype(str) + "-" + train["datetime-month"].astype(str)

print(train.shape)
train.info()
train[["datetime-year", "datetime-month", "datetime-year_month"]]
sns.barplot(data=train, x="datetime-year_month", y="count")
# 글자가 촘촘해서 보기 어려우니 크게 보고 싶을 때!

import matplotlib.pyplot as plt

plt.figure(figsize=(24,4))
sns.barplot(data=train, x="datetime-year_month", y="count")
plt.figure(figsize=(24,4))
sns.pointplot(data=train, x="datetime-hour", y="count")
plt.figure(figsize=(24,4))
sns.pointplot(data=train, x="datetime-hour", y="count", hue="workingday")
plt.figure(figsize=(24,4))
sns.pointplot(data=train, x="datetime-hour", y="count", hue="datetime-dayofweek")
# 특징 열의 이름을 리스트로 담아줍니다.
# feature_names = ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed",
#                  "datetime-year", "datetime-month", "datetime-day",
#                  "datetime-hour", "datetime-minute", "datetime-second"]

# feature_names = ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed",
#                  "datetime-year", "datetime-month", "datetime-hour"]

# 2018-01-24 Month를 빼보자
# feature_names = ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed",
#                  "datetime-year", "datetime-hour"]

# 2018-01-24 datetime-dayofweek을 넣어보자
feature_names = ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed",
                 "datetime-year", "datetime-hour", "datetime-dayofweek"]

# 리스트가 잘 들어갔는지 출력을 해보자.
feature_names
# 특징 열의 데이터를 x_train에 담는다.
x_train = train[feature_names]

# x_train의 데이터 크기를 보자.
print(x_train.shape)

# x_test가 정상적으로 들어갔는지 상위 5개만 출력해보자.
x_train.head()
# 특징 열의 데이터를 x_test에 담는다.
x_test = test[feature_names]

# x_test의 행렬의 크기를 보자.
print(x_test.shape)

# x_test가 정상적으로 들어갔는지 상위 5개만 출력해보자.
x_test.head()
# 레이블 데이터를 입력하자.
label_name = "count"
y_train = train[label_name]

print(y_train.shape)
y_train.head()
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeRegressor

# # model = DecisionTreeClassifier()

# # random_state를 줌으로써 이걸 돌리는 모든 사람들의 결과값이 일관되게 지정할 수 있다.
# #model = DecisionTreeClassifier(random_state=37)
# model = DecisionTreeRegressor(random_state=37)
# model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model
from sklearn.model_selection import cross_val_predict

y_predict = cross_val_predict(model, x_train, y_train, cv=20)

print(y_predict.shape)
y_predict
score = abs(y_train - y_predict).mean()

f"score(Mean Absolute Error)={score:.6f}"
# 학습을 시켜보자.
model.fit(x_train, y_train)
# x_test를 예측해보자.
predictions = model.predict(x_test)

# array 형태이다.
predictions
# 미리보기 하고 싶다면
predictions[:5]
submit = pd.read_csv("../input/sampleSubmission.csv")

print(submit.shape)
submit.head()
submit["count"] = predictions

print(submit.shape)
submit.head()
# x_test를 예측해보자.
predictions = model.predict(x_test)

# array 형태이다.
predictions
submit.to_csv("baseline-script.csv", index=False)
pd.read_csv("baseline-script.csv").head()