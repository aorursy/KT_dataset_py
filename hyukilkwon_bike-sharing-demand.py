# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



mpl.rcParams['axes.unicode_minus'] = False



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/bike-sharing-demand/train.csv", parse_dates=["datetime"])

train.shape
test = pd.read_csv("../input/bike-sharing-demand/test.csv", parse_dates=["datetime"])

test.shape
train["year"] = train["datetime"].dt.year

train["month"] = train["datetime"].dt.month

train["day"] = train["datetime"].dt.day

train["hour"] = train["datetime"].dt.hour

train["minute"] = train["datetime"].dt.minute

train["second"] = train["datetime"].dt.second

train["dayofweek"] = train["datetime"].dt.dayofweek

train.shape
test["year"] = test["datetime"].dt.year

test["month"] = test["datetime"].dt.month

test["day"] = test["datetime"].dt.day

test["hour"] = test["datetime"].dt.hour

test["minute"] = test["datetime"].dt.minute

test["second"] = test["datetime"].dt.second

test["dayofweek"] = test["datetime"].dt.dayofweek

test.shape
# widspeed has too many 0 value. => should be fixed

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
# separate wind speed as 0 and non 0

trainWind0 = train.loc[train['windspeed'] == 0]

trainWindNot0 = train.loc[train['windspeed'] != 0]

print(trainWind0.shape)

print(trainWindNot0.shape)

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
train = predict_windspeed(train)



fig, ax1 = plt.subplots()

fig.set_size_inches(18,6)



plt.sca(ax1)

plt.xticks(rotation=30, ha='right')

ax1.set(ylabel='Count',title="train windspeed")

sns.countplot(data=train, x="windspeed", ax=ax1)
categorical_feature_names = ["season","holiday","workingday","weather",

                             "dayofweek","month","year","hour"]



for var in categorical_feature_names:

    train[var] = train[var].astype("category")

    test[var] = test[var].astype("category")
feature_names = ["season", "weather", "temp", "atemp", "humidity", "windspeed",

                 "year", "hour", "dayofweek", "holiday", "workingday"]



feature_names
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

    

    predicted_values = np.array(predicted_values)

    actual_values = np.array(actual_values)

    

    log_predict = np.log(predicted_values + 1)

    log_actual = np.log(actual_values + 1)

    

    difference = log_predict - log_actual

    # difference = (log_predict - log_actual) ** 2

    difference = np.square(difference)

    

    mean_difference = difference.mean()

    

    score = np.sqrt(mean_difference)

    

    return score



rmsle_scorer = make_scorer(rmsle)

rmsle_scorer
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.ensemble import RandomForestRegressor



max_depth_list = []



model = RandomForestRegressor(n_estimators=100,

                              n_jobs=-1,

                              random_state=0)

model
%time score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)

score = score.mean()



print("Score= {0:.5f}".format(score))
model.fit(X_train, y_train)
predictions = model.predict(X_test)



print(predictions.shape)

predictions[0:10]
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