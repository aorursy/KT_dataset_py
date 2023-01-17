import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



#import os

#print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv", parse_dates=["datetime"])

test = pd.read_csv("../input/test.csv", parse_dates=["datetime"])

#train = pd.read_csv("../input/train.csv")

#test = pd.read_csv("../input/test.csv")
train.head(5)
test.head(5)
train["year"] = train["datetime"].dt.year

train["month"] = train["datetime"].dt.month

train["day"] = train["datetime"].dt.day

train["hour"] = train["datetime"].dt.hour

train["minute"] = train["datetime"].dt.minute

train["second"] = train["datetime"].dt.second

train["dayofweek"] = train["datetime"].dt.dayofweek



test["year"] = test["datetime"].dt.year

test["month"] = test["datetime"].dt.month

test["day"] = test["datetime"].dt.day

test["hour"] = test["datetime"].dt.hour

test["minute"] = test["datetime"].dt.minute

test["second"] = test["datetime"].dt.second

test["dayofweek"] = test["datetime"].dt.dayofweek
categorical_feature_names = ["season","holiday","workingday","weather","dayofweek","month","year","hour"]



for var in categorical_feature_names : 

    train[var] = train[var].astype("category")

    test[var] = test[var].astype("category")

feature_names = ["season","weather","temp","atemp","humidity","windspeed","year","hour","dayofweek","holiday","workingday"]
X_train = train[feature_names]

X_test = test[feature_names]

label_name = "count"

Y_train = train[label_name]
from sklearn.ensemble import RandomForestRegressor

max_depth_list = []

model = RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=0)

model.fit(X_train,Y_train)

predictions = model.predict(X_test)
submission = pd.read_csv("../input/sampleSubmission.csv")

submission["count"] = predictions

submission.to_csv("submission.csv",index=False)
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

sns.boxplot(data=train,y="count",orient="v",ax=axes[0][0])

sns.boxplot(data=train,y="count",x="season",orient="v",ax=axes[0][1])

sns.boxplot(data=train,y="count",x="hour",orient="v",ax=axes[1][0])

sns.boxplot(data=train,y="count",x="workingday",orient="v",ax=axes[1][1])

 

axes[0][0].set(ylabel='Count',title="Box Plot On Count")

axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")

axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")

axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")

 

corrMatt = train[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
fig,(ax1,ax2,ax3,ax4,ax5)= plt.subplots(nrows=5)

fig.set_size_inches(18,25)

sns.pointplot(data=train, x="hour", y="count", ax=ax1)

sns.pointplot(data=train, x="hour", y="count", hue="workingday", ax=ax2)

sns.pointplot(data=train, x="hour", y="count", hue="dayofweek", ax=ax3)

sns.pointplot(data=train, x="hour", y="count", hue="weather", ax=ax4)

sns.pointplot(data=train, x="hour", y="count", hue="season", ax=ax5)
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)

fig.set_size_inches(12, 5)

sns.regplot(x="temp", y="count", data=train,ax=ax1)

sns.regplot(x="windspeed", y="count", data=train,ax=ax2)

sns.regplot(x="humidity", y="count", data=train,ax=ax3)
from sklearn.ensemble import RandomForestClassifier

def predict_windspeed(data):

    dataWind0 = data.loc[data['windspeed'] == 0]

    dataWindNot0 = data.loc[data['windspeed'] != 0]

    wCol = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]

    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")

    rfModel_wind = RandomForestClassifier()

    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])

    wind0Values = rfModel_wind.predict(X = dataWind0[wCol])

    predictWind0 = dataWind0

    predictWindNot0 = dataWindNot0

    predictWind0["windspeed"] = wind0Values

    data = predictWindNot0.append(predictWind0)

    data["windspeed"] = data["windspeed"].astype("float")

    data.reset_index(inplace=True)

    data.drop('index', inplace=True, axis=1)

    return data