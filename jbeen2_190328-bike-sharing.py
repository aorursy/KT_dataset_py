%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
bike = pd.read_csv("../input/bikesharing/data_1.csv")

train = pd.read_csv("../input/bike-sharing-demand/train.csv")
print(bike.shape)
bike.head()
bike.info()
bike.isnull().sum()
figure, (ax1, ax2) = plt.subplots(ncols = 2)

figure.set_size_inches(12,4)



sns.boxplot(bike["cnt"], ax=ax1)

sns.distplot(bike["cnt"], ax=ax2)
bike[["casual", "registered", "cnt"]].sum()
figure, axes = plt.subplots(nrows=2, ncols = 2)

figure.set_size_inches(12,8)



sns.boxplot(bike["casual"], ax=axes[0][0])

sns.distplot(bike["casual"], ax=axes[0][1])

sns.boxplot(bike["registered"], ax=axes[1][0])

sns.distplot(bike["registered"], ax=axes[1][1])
sns.distplot(bike[["casual"]])

sns.distplot(bike[["registered"]])

sns.distplot(bike[["cnt"]])

plt.show()
from scipy.stats import ttest_ind

ttest_ind(bike[["casual"]], bike[["registered"]], equal_var = False)
def normalize(x):

    train[x] = (train[x] - train[x].min()) / (train[x].max() - train[x].min())

    return train[x].describe()
normalize("temp")
def year(x):

    if x == 0 :

        return 2011

    elif x == 1:

        return 2012



def season(x):

    if x in [12,1,2]:

        return 4

    elif x in [3,4,5]:

        return 1

    elif x in [6,7,8]:

        return 2

    elif x in [9,10,11]:

        return 3
bike["yr"] = bike["yr"].apply(year)

bike["season"] = bike["mnth"].apply(season)
bike.head()
# 연도, 월, 시간별 데이터와 count 사이의 관계



figure, (ax1, ax2, ax3) = plt.subplots(ncols=3)

figure.set_size_inches(18,5)



sns.barplot(data=bike, x="yr", y="cnt", ax=ax1)

sns.barplot(data=bike, x="mnth", y="cnt", ax=ax2)

sns.barplot(data=bike, x="hr", y="cnt", ax=ax3)
# 연도, 월, 시간별 데이터와 casual,registered 사이의 관계



figure, (ax1, ax2) = plt.subplots(ncols=2)

figure.set_size_inches(18,4)



sns.barplot(data=bike, x="hr", y="casual", ax=ax1)

sns.barplot(data=bike, x="hr", y="registered", ax=ax2)
# 계절, weather별 데이터와 count 사이의 관계



figure, axes = plt.subplots(nrows = 2, ncols=2)

figure.set_size_inches(12,12)



sns.barplot(data=bike, x='season', y="cnt", ax=axes[0][0])

sns.boxplot(data=bike, x="season", y="cnt", ax=axes[0][1])

sns.barplot(data=bike, x='weathersit', y="cnt", ax=axes[1][0])

sns.boxplot(data=bike, x='weathersit', y="cnt", ax=axes[1][1])
figure, axes = plt.subplots(nrows = 2, ncols=2)

figure.set_size_inches(12,12)



sns.barplot(data=bike, x='season', y="casual", ax=axes[0][0])

sns.barplot(data=bike, x="season", y="registered", ax=axes[0][1])

sns.barplot(data=bike, x='weathersit', y="casual", ax=axes[1][0])

sns.barplot(data=bike, x='weathersit', y="registered", ax=axes[1][1])
# holiday, weekday, workingday와 자전거 대여량 비교



figure, axes = plt.subplots(ncols=3)

figure.set_size_inches(18, 6)



sns.barplot(data=bike, x='holiday', y="cnt", ax=axes[0])

sns.barplot(data=bike, x="weekday", y="cnt", ax=axes[2])

sns.barplot(data=bike, x='workingday', y="cnt", ax=axes[1])
# 온도, 체감온도, 습도, 풍속

bike[["temp", "atemp", "hum", "windspeed"]].describe()
sns.jointplot(x="temp", y="cnt", data = bike, kind="kde")
sns.jointplot(x="atemp", y="cnt", data = bike, kind="kde")
sns.jointplot(x="hum", y="cnt", data = bike, kind="kde")
sns.jointplot(x="windspeed", y="cnt", data = bike, kind="kde")
# 분포를 보면 풍속에 0값이 많아보이므로 0값을 예측하여 대체해주기로 함



bikewind0 = bike.loc[bike["windspeed"]==0]

bikewindnot0 = bike.loc[bike["windspeed"]!=0]

print(bikewind0.shape, bikewindnot0.shape)
from sklearn.neighbors import KNeighborsClassifier



def predict_wind(data):

    datawind0 = data.loc[data["windspeed"] == 0]

    datawindnot0 = data.loc[data["windspeed"] != 0]

    

    wcol = ["season", "mnth", "weathersit", "temp", "atemp", "hum"]

    

    datawindnot0["windspeed"] = datawindnot0["windspeed"].astype("str")

    

    rfmodel = KNeighborsClassifier()

    rfmodel.fit(datawindnot0[wcol], datawindnot0["windspeed"])

    

    wind0values = rfmodel.predict(X=datawind0[wcol])

    

    predictwind0 = datawind0

    predictwindnot0 = datawindnot0

    

    predictwind0["windspeed"] = wind0values

    

    data = predictwindnot0.append(predictwind0)

    data["windspeed"] = data["windspeed"].astype("float")

    

    data.reset_index(inplace = True)

    data.drop("index", inplace = True, axis=1)

    

    return data
w_bike = predict_wind(bike)

w_bike["windspeed"].describe()
bike["windspeed"] = w_bike["windspeed"]



bikewind0 = bike.loc[bike["windspeed"]==0]

bikewindnot0 = bike.loc[bike["windspeed"]!=0]

print(bikewind0.shape, bikewindnot0.shape)
sns.jointplot(x="windspeed", y="cnt", data = bike, kind="kde")
bike.head()
# temp, atemp, hum, windspeed 각각 변수들간 상관관계 있는지 파악



weather = bike[["temp","atemp","hum","windspeed","cnt"]]

plt.figure(figsize=(5,5))

sns.heatmap(weather.corr(), annot=True)

plt.show()
bike.columns
# 시간에 따른 대여량



fig, axes = plt.subplots(nrows = 6)

fig.set_size_inches(18,30)



sns.pointplot(data=bike, x="hr", y="cnt", ax=axes[0])

sns.pointplot(data=bike, x="hr", y="cnt", hue = "workingday", ax=axes[1])

sns.pointplot(data=bike, x="hr", y="cnt", hue = "holiday", ax=axes[2])

sns.pointplot(data=bike, x="hr", y="cnt", hue = "weekday", ax=axes[3])

sns.pointplot(data=bike, x="hr", y="cnt", hue = "weathersit", ax=axes[4])

sns.pointplot(data=bike, x="hr", y="cnt", hue = "season", ax=axes[5])
bike[["holiday", "weekday", "workingday"]].corr()
# workingday의 경우 weekday(1~5)의 분포와 비슷해 보이는데, 

# 이는 상관관계에서 직접적으로 나타나지 않음 -> 변수 변환을 하고 따져봐야 하기 때문



def workingday(x):

    if x in [1,2,3,4,5]:

        return 1

    elif x in [0,6]:

        return 0



bike["dummywork"] = bike.weekday.apply(workingday)

bike.head()
bike[["workingday", "dummywork"]].corr()
bike_no_outlier = bike.copy()

bike_no_outlier = bike[np.abs(bike["cnt"] - bike["cnt"].mean()) <= 3*bike["cnt"].std()]

bike_no_outlier.shape[0] / bike.shape[0]
fig, axes = plt.subplots(nrows=2, ncols=2)

fig.set_size_inches(12,10)



sns.distplot(bike["cnt"], ax=axes[0][0])

stats.probplot(bike["cnt"], dist="norm", fit=True, plot=axes[0][1])

sns.distplot(np.log(bike_no_outlier["cnt"]), ax=axes[1][0])

stats.probplot(np.log1p(bike_no_outlier["cnt"]), dist="norm", fit=True, plot=axes[1][1])
import statsmodels.api as sm 



X = bike[["temp", "hum", "windspeed"]]

y = np.log1p(bike[["cnt"]])



model2 = sm.OLS(y, X)

result = model2.fit()

print(result.summary())
bike_no_outlier.columns
dropfeatures = ['instant', 'dteday', 'atemp', 'weekday', 'dummywork', 'casual', 'registered']

dummyfeatures = ['season', 'yr', 'mnth', 'hr', 'weathersit']
for each in dummyfeatures:

    dummies = pd.get_dummies(bike_no_outlier[each], prefix=each, drop_first = False)

    bike_no_outlier = pd.concat([bike_no_outlier, dummies], axis=1)

    

bike_no_outlier = bike_no_outlier.drop(dropfeatures, axis=1)

bike_no_outlier.head()
bike_no_outlier.columns
bike_no_outlier = bike_no_outlier.drop(dummyfeatures, axis=1)

bike_no_outlier.columns
bike_no_outlier["cnt"] = np.log1p(bike_no_outlier["cnt"])

bike_no_outlier.head()
from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
x_train,x_test,y_train,y_test=train_test_split(bike_no_outlier.drop('cnt',axis=1),bike_no_outlier['cnt'],test_size=0.3)
def rmsle(pred, actual):

    pred_val = np.array(pred)

    act_val = np.array(actual)

    

    log_predict = np.log(pred_val + 1)

    log_actual = np.log(act_val + 1)

    cal = (log_predict - log_actual) ** 2 

    

    return np.sqrt(np.mean(cal))
model1 = LinearRegression(normalize=True)

model1.fit(x_train, y_train)

print("Accuracy: {}, RMSLE: {}".format(model1.score(x_test, y_test), rmsle(y_test, model1.predict(x_test))))
model2 = DecisionTreeRegressor()

model2.fit(x_train, y_train)

print("Accuracy: {}, RMSLE: {}".format(model2.score(x_test, y_test), rmsle(y_test, model2.predict(x_test))))
model3 = RandomForestRegressor()

model3.fit(x_train, y_train)

print("Accuracy: {}, RMSLE: {}".format(model3.score(x_test, y_test), rmsle(y_test, model3.predict(x_test))))
model4 = KNeighborsRegressor()

model4.fit(x_train, y_train)

print("Accuracy: {}, RMSLE: {}".format(model4.score(x_test, y_test), rmsle(y_test, model4.predict(x_test))))