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
train_data = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv', parse_dates=['datetime'])

test_data = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv', parse_dates=["datetime"])

sample_data = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')



y = train_data["count"]



train_data["year"] = train_data["datetime"].dt.year

train_data["month"] = train_data["datetime"].dt.month

train_data["day"] = train_data["datetime"].dt.dayofweek

train_data["hour"] = train_data["datetime"].dt.hour

train_data = train_data.drop(["datetime", "casual", "registered", "count"], axis=1)



test_data["year"] = test_data["datetime"].dt.year

test_data["month"] = test_data["datetime"].dt.month

test_data["day"] = test_data["datetime"].dt.dayofweek

test_data["hour"] = test_data["datetime"].dt.hour

test_data = test_data.drop(["datetime"], axis=1)
train_data.head()
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier

humidity_train = train_data[train_data["humidity"] > 0]

humidity_ans = humidity_train["humidity"]

humidity_train = humidity_train.drop(["humidity"], axis=1)

humidity_model = RandomForestClassifier()

humidity_model.fit(humidity_train, humidity_ans)



humidity_train.head()
index_data = train_data[train_data["humidity"] == 0].index

for i in range(len(index_data)):

    tmp_data = train_data.loc[[index_data[i]]]

    tmp_data = tmp_data.drop(["humidity"], axis=1)

    pred_humidity = humidity_model.predict(tmp_data)

    train_data["humidity"][index_data[i]] = pred_humidity
train_data.head()
wind_train = train_data[train_data["windspeed"] > 0]

wind_ans = wind_train["windspeed"]

wind_train = wind_train.drop(["windspeed"], axis=1)

windmodel = GradientBoostingRegressor()

windmodel.fit(wind_train, wind_ans)

wind_train.head()
train_data[train_data["windspeed"] == 0].index
#데이터 검색 후 모델 활용을 통한 풍속 빈값 업데이트

index_data = train_data[train_data["windspeed"] == 0].index



for i in range(len(index_data)):

    tmp_data = train_data.loc[[index_data[i]]]

    tmp_data = tmp_data.drop(["windspeed"], axis=1)

    pred_wind = windmodel.predict(tmp_data)

    train_data["windspeed"][index_data[i]] = pred_wind
train_data.head()
test_data.head()
index_data = test_data[test_data["humidity"] == 0].index

for i in range(len(index_data)):

    tmp_data = test_data.loc[[index_data[i]]]

    tmp_data = tmp_data.drop(["humidity"], axis=1)

    pred_humidity = humidity_model.predict(tmp_data)

    test_data["humidity"][index_data[i]] = pred_humidity
#데이터 검색 후 모델 활용을 통한 풍속 빈값 업데이트

index_data = test_data[test_data["windspeed"] == 0].index



for i in range(len(index_data)):

    tmp_data = test_data.loc[[index_data[i]]]

    tmp_data = tmp_data.drop(["windspeed"], axis=1)

    pred_wind = windmodel.predict(tmp_data)

    test_data["windspeed"][index_data[i]] = pred_wind
final_model = RandomForestClassifier()

final_model.fit(train_data, y)

ans = final_model.predict(test_data)

sample_data["count"] = ans

sample_data.head()
sample_data.to_csv('analysis_final.csv', index=False)