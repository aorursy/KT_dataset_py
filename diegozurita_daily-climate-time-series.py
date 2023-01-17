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
import seaborn as sns

import matplotlib.pyplot as plt

import scipy

from sklearn.metrics import mean_squared_error



sns.set()
daily_climate_train_df = pd.read_csv("/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv", parse_dates=["date"]) 

daily_climate_test_df = pd.read_csv("/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTest.csv", parse_dates=["date"]) 



daily_climate_train_df.shape, daily_climate_test_df.shape
daily_climate_train_df.date.min(), daily_climate_train_df.date.max()
daily_climate_test_df.date.min(), daily_climate_test_df.date.max()
daily_climate_train_df.sort_values("date", inplace=True)
daily_climate_train_df.count()
def custom_train_test_split(data, size=1200):

    explain_columns = ["date", "humidity", "wind_speed", "meanpressure"]   

    y_column = "meantemp"



    X_train = data[explain_columns].iloc[:size]

    X_test = data[explain_columns].iloc[size:]

    y_train = data[y_column].iloc[:size]

    y_test = data[y_column].iloc[size:]

    

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = custom_train_test_split(daily_climate_train_df)
_ = plt.figure(figsize=(19, 8))



_ = plt.subplot(131)

_ = plt.hist(X_train["humidity"], bins=30, density=True)

_ = plt.xlabel("Humidity")



_ = plt.subplot(132)

_ = plt.hist(X_train["wind_speed"], bins=30, density=True)

_ = plt.xlabel("Wind speed")





_ = plt.subplot(133)

_ = plt.hist(X_train["meanpressure"], density=True)

_ = plt.xlabel("Mean pressure")



plt.show()
_ = plt.figure(figsize=(16, 4))

sns.lineplot(x="date", y="meanpressure", data=X_train)

plt.show()
X_train[X_train["date"] < "2016-01-01"]["meanpressure"].max() - X_train[X_train["date"] < "2016-01-01"]["meanpressure"].min()
scipy.stats.pearsonr(X_train["meanpressure"], y_train)
_ = plt.scatter(X_train["meanpressure"], y_train)

_ = plt.xlabel("meanpressure")

_ = plt.ylabel("Mean temp")



plt.show()
def custom_train_test_split_v2(data, size=1200):

    explain_columns = ["date", "humidity", "wind_speed"]   

    y_column = "meantemp"



    X_train = data[explain_columns].iloc[:size]

    X_test = data[explain_columns].iloc[size:]

    y_train = data[y_column].iloc[:size]

    y_test = data[y_column].iloc[size:]

    

    y_test.index = np.arange(len(y_test))

    y_train.index = np.arange(len(y_train))

    

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = custom_train_test_split_v2(daily_climate_train_df)
_ = plt.figure(figsize=(18, 9))



_ = plt.plot(X_train["date"], X_train["humidity"], label="Humidity")

_ = plt.plot(X_train["date"], X_train["wind_speed"], label="Wind speed")

_ = plt.plot(X_train["date"], y_train, label="Mean temp")



# the vertical lines show a new year start

_ = plt.axvline(x="2013-01-01", c="r", linewidth=3.0)

_ = plt.axvline(x="2014-01-01", c="r", linewidth=3.0)

_ = plt.axvline(x="2015-01-01", c="r", linewidth=3.0)

_ = plt.axvline(x="2016-01-01", c="r", linewidth=3.0)



_ = plt.legend()



plt.show()
def get_mean_temp_for_n_days (data, days=7):

    mean_temp_last_n_days = np.empty(len(data))



    for i in range(len(data)):

        if i < days: 

            mean_temp_last_n_days[i] = 0

        else:

            mean_temp_last_n_days[i] = data[i - days: i].mean()



    return mean_temp_last_n_days
_ = plt.figure(figsize=(18, 9))



#_ = plt.plot(X_train["date"], X_train["humidity"], label="Humidity")

_ = plt.plot(X_train["date"], get_mean_temp_for_n_days(y_train, days=7) + 10, label="Last 7 days")

_ = plt.plot(X_train["date"], get_mean_temp_for_n_days(y_train, days=2) + 5, label="Last 2 days")

_ = plt.plot(X_train["date"], y_train, label="Mean temp")



# the vertical lines show a new year start

_ = plt.axvline(x="2013-01-01", c="r", linewidth=3.0)

_ = plt.axvline(x="2014-01-01", c="r", linewidth=3.0)

_ = plt.axvline(x="2015-01-01", c="r", linewidth=3.0)

_ = plt.axvline(x="2016-01-01", c="r", linewidth=3.0)



_ = plt.legend()



plt.show()
# start with 100 days

n = 100

errors = np.empty(n)



for i in range(n):

    y_pred = get_mean_temp_for_n_days(y_train, days = i + 2)

    errors[i] = mean_squared_error(y_train, y_pred)

    

_ = plt.plot(np.arange(n) + 2, errors)

_ = plt.xlabel("Days on mean")

_ = plt.ylabel("Error")



plt.show()
# start with 100 days

n = 100

errors = np.empty(n)



for i in range(n):

    y_pred = get_mean_temp_for_n_days(y_train, days = i + 2)

    errors[i] = mean_squared_error(y_train[i + 2: ], y_pred[i + 2: ])

    

_ = plt.plot(np.arange(n) + 2, errors)

_ = plt.xlabel("Days on mean")

_ = plt.ylabel("Error")



plt.show()
_ = plt.figure(figsize=(20, 9))



x = daily_climate_train_df["date"]

y_real = daily_climate_train_df["meantemp"]

y_pred = get_mean_temp_for_n_days(y_real, days=2)



mean_error = np.quantile(np.abs(y_pred - y_real), 0.95)



print("Expected error: ", mean_error)



_ = plt.plot(x, y_real, label="True")

_ = plt.plot(x, y_pred, label="Pred")

_ = plt.fill_between(x, y1=y_pred + mean_error, y2=y_pred - mean_error, color="orange", alpha=0.2)



_ = plt.xlabel("Date")

_ = plt.ylabel("mean temperature")



_ = plt.legend()



plt.show()
_ = plt.figure(figsize=(20, 9))



x = daily_climate_train_df["date"][:400]

y_real = daily_climate_train_df["meantemp"][:400]

y_pred = get_mean_temp_for_n_days(y_real, days=2)



mean_error = np.quantile(np.abs(y_pred - y_real), 0.95)



print("Expected error: ", mean_error)



_ = plt.plot(x, y_real, label="True")

_ = plt.plot(x, y_pred, label="Pred")

_ = plt.fill_between(x, y1=y_pred + mean_error, y2=y_pred - mean_error, color="orange", alpha=0.2)



_ = plt.xlabel("Date")

_ = plt.ylabel("mean temperature")



_ = plt.legend()



plt.show()