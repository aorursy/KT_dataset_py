import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 노트북 안에 그래프를 그리기 위해
%matplotlib inline

# 그래프에서 격자로 숫자 범위가 눈에 잘 띄도록 ggplot 스타일을 사용
plt.style.use('ggplot')

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", parse_dates=["datetime"])
#parse_dates : date 형식으로 불러오기
train.shape
test = pd.read_csv("../input/test.csv", parse_dates=["datetime"])
test.shape
# test에 없는 변수가 세개나 됨
test.info()
# casual, register, count 세개의 변수가 없는것을 확인할 수 있음
train.info() # 결측치는 없음
train.head()
train.describe() # describe 하면 datetime 은 안나오네..!
# season, holiday, workingday, weather 는 범주형이 나을듯
# datetime 을 쪼개 의미있는 변수만 추려보겠음
# https://datascienceschool.net/view-notebook/465066ac92ef4da3b0aba32f76d9750a/ 참고
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second
### 플롯 그려보기
figure, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3) 
# par(mfrow=c(2,3)) 같은것, axis를 정해주는 이유는 그래프의 위치 정해주려고
figure.set_size_inches(18,8)
sns.barplot(data=train, x="year", y="count", ax=ax1)
sns.barplot(data=train, x="month", y="count", ax=ax2)
sns.barplot(data=train, x="day", y="count", ax=ax3)
sns.barplot(data=train, x="hour", y="count", ax=ax4)
sns.barplot(data=train, x="minute", y="count", ax=ax5)
sns.barplot(data=train, x="second", y="count", ax=ax6)
# 2012년의 대여량이 많고, 월별로는 여름 가을에 가장 많으며
# 시간대로는 아침과 저녁이 월등히 많은 것을 볼 수 있음
sns.distplot(train["count"]) # count의 분포가 정규분포가 아니기 때문에 예측이 어려울 수 있음
# outlier 제거
train1 = train[np.abs(train["count"] - train["count"].mean()) <= (3*train["count"].std())]
print(train.shape, train1.shape) # 150 개정도 제거
sns.distplot(np.log(train1["count"])) # 로그변환하니 이전보다 정규분포를 보임
figure, ((ax1,ax2,ax3,ax4), (ax5,ax6,ax7,ax8)) = plt.subplots(nrows=2, ncols=4) 
# par(mfrow=c(2,3)) 같은것, axis를 정해주는 이유는 그래프의 위치 정해주려고
figure.set_size_inches(16,8)
sns.barplot(data=train, x="season", y="count", ax=ax1)
sns.barplot(data=train, x="holiday", y="count", ax=ax2)
sns.barplot(data=train, x="workingday", y="count", ax=ax3)
sns.barplot(data=train, x="weather", y="count", ax=ax4)
sns.barplot(data=train, x="temp", y="count", ax=ax5)
sns.barplot(data=train, x="atemp", y="count", ax=ax6)
sns.barplot(data=train, x="humidity", y="count", ax=ax7)
sns.barplot(data=train, x="windspeed", y="count", ax=ax8)

# season 은 앞에서 만든 month 와 비슷한 결과
# weather 는 맑을수록 높은 대여량을 보임
# temp 와 atemp 는 대체로 기온이 높아짐에 따라 대여량이 증가함
# humidity 가 높아짐에 따라 대여량이 감소함
# 풍속은 너무 높을때가 아니면 대여량은 거의 비슷함
# 방법1 : test에 없는 casual 과 registered 를 아예 없애버림
# train.drop(["datetime","day","minute","second","casual","registered"], axis=1)
# 하지만
corrMatt = train[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]]
corrMatt = corrMatt.corr()
print(corrMatt)
# 상관관계를 보니 registered 와 casual 의 상관관계가 매우 높음을 알 수 있다.
# 따라서
# 방법2 : test에 없는 casual과 registered 값을 예측해 채워준 다음, 최종 count를 예측해 보겠다
# 우선 1차적으로 쓸모없는 변수부터 지워줌
train = train.drop(['day', 'minute', 'second'],axis=1)
train.head()
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['hour'] = test['datetime'].dt.hour
test.head()
test.shape
# register 와 casual은 예측 후 채워줄 것임
from sklearn.ensemble import RandomForestRegressor
f_columns = ['season','weather','humidity','month','temp','year','hour','atemp']
X_train = train[f_columns]
y_r_label = train['registered']
y_c_label = train['casual']
print(X_train.shape)
print(y_r_label.shape)
print(y_c_label.shape)

X_test = test[f_columns]
X_test.shape
rfModel = RandomForestRegressor()
rfModel.fit(X_train, y_r_label)
testRegisteredValues = rfModel.predict(X_test)
test['registered'] = testRegisteredValues
rfModel.fit(X_train, y_c_label)
testCasualValues = rfModel.predict(X_test)
test['casual'] = testCasualValues
# test 와 train 데이터 합침
data = train.append(test)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)
categoricalFeatureNames = ['season','holiday','workingday','weather','year','hour']
for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")
X_train = data[pd.notnull(data['count'])].sort_values(by=['datetime'])
X_test = data[~pd.notnull(data['count'])].sort_values(by=['datetime'])
datetimecol = X_test['datetime']
y_train = X_train['count']
dropFeatures = ['count','datetime','month']# month는 season 과 의미하는 바가 동일
X_train  = X_train.drop(dropFeatures,axis=1)
X_test  = X_test.drop(dropFeatures,axis=1)
rfModel = RandomForestRegressor(n_estimators=100)

y_train_log = np.log1p(y_train) # 앞에서 봤던것 처럼 count 변수는 로그변환이 필요함
# log1p 는 inf 값 방지하는 함수
# http://rfriend.tistory.com/295 참고
rfModel.fit(X_train, y_train_log)
preds = rfModel.predict(X_train) 
# predict 까지는 했는데 잘 예측되었는지 확인하는 척도가 필요, 여기서는 rmsle 값을 사용
# rmsle 란 : https://www.quora.com/What-is-the-difference-between-an-RMSE-and-RMSLE-logarithmic-error-and-does-a-high-RMSE-imply-low-RMSLE
from sklearn.metrics import make_scorer

def rmsle(predicted_values, actual_values, convertExp=True):

    if convertExp:
        predicted_values = np.exp(predicted_values),
        actual_values = np.exp(actual_values)
        
    # 넘파이로 배열 형태로 바꿔준다.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다.
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.
    difference = log_predict - log_actual
    difference = np.square(difference)
    
    # 평균을 낸다.
    mean_difference = difference.mean()
    
    # 다시 루트를 씌운다.
    score = np.sqrt(mean_difference)
    
    return score
score = rmsle(np.exp(y_train_log),np.exp(preds),False)
print ("RMSLE Value For Random Forest: ",score)
predsTest = rfModel.predict(X_test)
# 아까 로그변환했으니 제출시 지수변환해서 제출
submission = pd.read_csv("../input/sampleSubmission.csv")
submission

submission["count"] = np.round(np.exp(predsTest),0)

print(submission.shape)
submission.head()
submission.to_csv("Score_{0:.5f}_submission.csv".format(score), index=False)
