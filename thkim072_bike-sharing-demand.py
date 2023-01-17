import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats

# seaborn 라이브러리 세팅
plt.style.use('ggplot')    # matplot 기본 그림 말고 격자로 숫자 범위가 눈에 잘 뛰도록 ggplot스타일 사용
sns.set(font_scale=2.5)    # 폰트 사이즈 2.5로 고정

# null 데이터를 시각화하여 보여주는 라이브러리
import missingno as msno   

# 오류 무시하는 코드 
import warnings
warnings.filterwarnings('ignore')

# matplot 라이브러리 사용해 시각화한 뒤 show했을 때 새로운 창이 아닌 노트북에서 바로 확인 가능하도록
%matplotlib inline

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False
train = pd.read_csv('../input/bike-sharing-demand/train.csv',parse_dates=['datetime'])
test = pd.read_csv('../input/bike-sharing-demand/test.csv',parse_dates=['datetime'])
copy_train = train.copy()
copy_test = test.copy()
copy_train.head()
# windspeed는 대부분이 0으로 되어있는데 이는 데이터가 없는 부분을 0으로 채운 것 같음 -> 피처 엔지니어링 필요
copy_train.shape
copy_train.info()
copy_train.isnull().sum()
copy_train['year']=copy_train['datetime'].dt.year
copy_train['month']=copy_train['datetime'].dt.month
copy_train['day']=copy_train['datetime'].dt.day
copy_train['hour']=copy_train['datetime'].dt.hour
copy_train['minute']=copy_train['datetime'].dt.minute
copy_train['second']=copy_train['datetime'].dt.second
copy_test['year']=copy_test['datetime'].dt.year
copy_test['month']=copy_test['datetime'].dt.month
copy_test['day']=copy_test['datetime'].dt.day
copy_test['hour']=copy_test['datetime'].dt.hour
copy_test['minute']=copy_test['datetime'].dt.minute
copy_test['second']=copy_test['datetime'].dt.second
copy_train.head()
copy_test.head()
# (ax1,ax2,ax3), (ax4,ax5,ax6)로 alias 지정하여 시각화
figure, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(18,8)

sns.barplot(data=copy_train, x="year", y="count", ax=ax1)
sns.barplot(data=copy_train, x="month", y="count", ax=ax2)
sns.barplot(data=copy_train, x="day", y="count", ax=ax3)
sns.barplot(data=copy_train, x="hour", y="count", ax=ax4)
sns.barplot(data=copy_train, x="minute", y="count", ax=ax5)
sns.barplot(data=copy_train, x="second", y="count", ax=ax6)

ax1.set(ylabel='Count',title="연도별 대여량")
ax2.set(xlabel='month',title="월별 대여량")
ax3.set(xlabel='day', title="일별 대여량")
ax4.set(xlabel='hour', title="시간별 대여량")

# 연도별 대여량은 2011년 보다 2012년이 더 많음
# 월별 대여량은 6월에 가장 많고 7~10월도 대여량이 많음. 그리고 1월에 가장 적음
# 일별 대여량은 1일부터 19일까지만 있고 나머지 날짜는 test.csv에 있음. 그래서 이 데이터는 피처로 사용하면 안됨
# 시간별 대여량을 보면 출퇴근 시간에 대여량이 많은 것 같음. 하지만 주말과 나누어 볼 필요가 있음
# 분, 초도 다 0이기 때문에 의미가 없음
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)

sns.boxplot(data=copy_train,y="count",orient="v",ax=axes[0][0])
sns.boxplot(data=copy_train,y="count",x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=copy_train,y="count",x="hour",orient="v",ax=axes[1][0])
sns.boxplot(data=copy_train,y="count",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="대여량")
axes[0][1].set(xlabel='Season', ylabel='Count',title="계절별 대여량")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="시간별 대여량")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="근무일 여부에 따른 대여량")

# 대여량만 보면 특정 구간에 몰려있음
# 계절별 대여량을 보면 봄이 가장 적고, 여름과 가을이 가장 많음
# 시간별 대여량은 위에서 그린 내용과 같음
# 근무일 여부에 따른 큰 차이는 없음
copy_train["dayofweek"] = copy_train["datetime"].dt.dayofweek
copy_test["dayofweek"] = copy_test["datetime"].dt.dayofweek
copy_train["dayofweek"].value_counts()
# 요일은 0~6까지 골고루 분포함을 알 수 있음
fig,(ax1,ax2,ax3,ax4,ax5)= plt.subplots(nrows=5)
fig.set_size_inches(18,25)

sns.pointplot(data=copy_train, x="hour", y="count", ax=ax1)   # 출퇴근시간에 많이 대여함
sns.pointplot(data=copy_train, x="hour", y="count", hue="workingday", ax=ax2)  # workingday로 구분해서 보면 출퇴근시간 뿐만 아니라 휴일에는 점심,오후에도 많이 대여 
sns.pointplot(data=copy_train, x="hour", y="count", hue="dayofweek", ax=ax3)   # 5,6인 토,일의 대여가 많음
sns.pointplot(data=copy_train, x="hour", y="count", hue="weather", ax=ax4)     # 날씨좋을 때 많이 빌림
sns.pointplot(data=copy_train, x="hour", y="count", hue="season", ax=ax5)      # 봄에 가장 적게 빌림
corrMatt = copy_train[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]]
corrMatt = corrMatt.corr()
print(corrMatt)

mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)

# 온도, 습도, 풍속은 거의 연관관계가 없음
# 대여량과 가장 연관이 높은 건 등록된 사용자 registered, 풍속과 연관이 높은 건 등록되지 않은 사용자 casual이지만 test데이터에는 두 칼럼이 피처로 사용할 수 없음
# atemp와 temp는 체감온도와 온도로 0.98로 상관관계가 너무 높아 피처로 사용하기에 적합하지 않을 수 있음 -> 회귀 계수는 상관관계가 높으면 분산이 커져 오류에 민감(다중공선성)
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
fig.set_size_inches(12, 5)

sns.regplot(x="temp", y="count", data=copy_train,ax=ax1)
sns.regplot(x="windspeed", y="count", data=copy_train,ax=ax2)
sns.regplot(x="humidity", y="count", data=copy_train,ax=ax3)

# 풍속의 경우 0에 숫자가 많이 몰려 있는 것으로 보임. 아마도 관측되지 않은 수치에 대해 0으로 기록한 것 같음
# 습도도 0과 100에 몰려있음
fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(18,10)

# train
plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(ylabel='Count',title="train windspeed")
sns.countplot(data=copy_train, x="windspeed", ax=axes[0])

# test
plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[1].set(ylabel='Count',title="test windspeed")
sns.countplot(data=copy_test, x="windspeed", ax=axes[1])
copy_train.loc[copy_train["windspeed"] == 0, "windspeed"] = copy_train["windspeed"].mean()
copy_test.loc[copy_test["windspeed"] == 0, "windspeed"] = copy_test["windspeed"].mean()
fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(18,10)

# train
plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(ylabel='Count',title="train windspeed")
sns.countplot(data=copy_train, x="windspeed", ax=axes[0])

# test
plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[1].set(ylabel='Count',title="test windspeed")
sns.countplot(data=copy_test, x="windspeed", ax=axes[1])
feature_names = ["season", "weather", "temp", "atemp", "humidity", "windspeed", "year", "hour", "dayofweek", "holiday", "workingday"]
feature_names
copy_train = copy_train[feature_names]
copy_test = copy_test[feature_names]
copy_train
copy_test
X = copy_train            # count없는 피처들
Y = train['count']        # count만 있는 피처
X
Y
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=13)
from sklearn.metrics import make_scorer

def rmsle(predicted_values, actual_values, convertExp=True):

    if convertExp:
        predicted_values = np.exp(predicted_values),
        actual_values = np.exp(actual_values)
        
    # 넘파이로 배열 형태로 바꿔줌
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    # 예측값과 실제 값에 1을 더하고 로그를 씌워줌
    # 값이 0일 수도 있어서 로그를 취했을 때 마이너스 무한대가 될 수도 있기 때문에 1을 더해 줌
    # 로그를 씌워주는 것은 정규분포로 만들어주기 위해
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱
    difference = log_predict - log_actual
    difference = np.square(difference)
    
    # 평균
    mean_difference = difference.mean()
    
    # 다시 루트를 씌움
    score = np.sqrt(mean_difference)
    
    return score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

lr = LinearRegression()  

# count값의 최저 값과 최고 값의 낙폭이 너무 커서 log를 취하지 않으면 inf로 결과가 나옴
# np.log1p는 np.log(1+x)와 동일
# 만약 어떤 x값이 0인데 이를 log하게되면 (-)무한대로 수렴하기 때문에 np.log1p를 사용"""
y_train_log = np.log1p(y_train)

# 학습
lr.fit(X_train, y_train_log)

# 예측
lr_pred = lr.predict(X_train)

# 평가
# np.exp: pred로 나온 값은 이미 log를 한 값이라 원래 모델에는 log를 하지 않은 원래의 값을 넣기 위함
print ("RMSLE : ", rmsle(np.exp(y_train_log),np.exp(lr_pred), False))  
print('절편 값:',lr.intercept_) 
print('회귀 계수값:', np.round(lr.coef_, 1)) 
# lr.coef_는 회귀 계수 값만 나오므로 피처별 회귀 계수 값으로 다시 매핑
coef = pd.Series(data=np.round(lr.coef_, 1), index=X.columns )  # 데이터는 회귀 계수 값, 인덱스는 X의 칼럼명
coef.sort_values(ascending=False)  # 내림차순 정렬
ridge = Ridge()
parameters = {'max_iter':[3000],'alpha':[0.001,0.01,0.1,1,10,100,1000]}
rmsle_scorer = metrics.make_scorer(rmsle,greater_is_better=False)
grid_ridge = GridSearchCV(ridge,parameters,scoring=rmsle_scorer,cv=5)

grid_ridge.fit(X_train,y_train_log)
pred = grid_ridge.predict(X_train)
print(grid_ridge.best_params_)
print('RMSLE Value for Ridge Regression {}'.format(rmsle(np.exp(y_train_log),np.exp(pred),False)))
df = pd.DataFrame(grid_ridge.cv_results_)
df.head()
lasso = Lasso()
parameters = {'max_iter':[3000],'alpha':[0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}

grid_lasso = GridSearchCV(lasso,parameters,scoring = rmsle_scorer,cv=5)
y_train_log = np.log1p(y_train)
grid_lasso.fit( X_train , y_train_log )
pred = grid_lasso.predict(X_train)
print (grid_lasso.best_params_)
print ("RMSLE : ",rmsle(np.exp(y_train_log),np.exp(pred),False))
df = pd.DataFrame(grid_lasso.cv_results_)
df.head()
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)

y_train_log = np.log1p(y_train)
rf.fit(X_train, y_train_log)

pred = rf.predict(X_train)
score = rmsle(np.exp(y_train_log),np.exp(pred),False)
print ("RMSLE Value For Random Forest: ",score)
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000, alpha=0.01);

y_train_log = np.log1p(y_train)
gbm.fit(X_train, y_train_log)

pred = gbm.predict(X_train)
score = rmsle(np.exp(y_train_log),np.exp(pred),False)
print ("RMSLE Value For Gradient Boost: ", score)
submission = pd.read_csv("../input/bike-sharing-demand/sampleSubmission.csv")
submission
prediction = gbm.predict(copy_test)  # 실제 예측
prediction
submission['count'] = np.exp(prediction)  # count에 내가 실제로 예측한걸 저장
submission.head()
submission.to_csv('submission.csv', index = False)  # 캐글 커널 서버에 csv파일 저장
