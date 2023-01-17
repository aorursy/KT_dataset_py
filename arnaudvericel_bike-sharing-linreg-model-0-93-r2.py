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
data = pd.read_csv("/kaggle/input/bike-sharing-dataset/hour.csv")
data.head(3)
data.info()
# hours spread from 00 (midnight) to 23 (11 pm)

data.hr.value_counts()
import matplotlib

import matplotlib.pyplot as plt

import plotly_express as px

import seaborn as sns

import math
# We already have relevant info on date with yr, month and hour

# and we want only the total count

# also instant is the irrelevant for prediction

pre_dropped = ["dteday", "casual", "registered", "instant"]

data_prep = data.drop(pre_dropped, axis=1)

data_prep.isnull().sum() # no missing data
data_prep.columns
# let's plot the distributions of the different columns

data_prep.hist(rwidth=0.9, figsize=(20, 20))

plt.tight_layout()

plt.show()
# there's a few numerical columns

data_prep.head(10)
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)

plt.title("Demand = f(Temperature)")

plt.scatter(x=data_prep.temp, y=data_prep.cnt, s=2, c="magenta")

##

plt.subplot(2, 2, 2)

plt.title("Demand = f(Feeled Temperature)")

plt.scatter(x=data_prep.atemp, y=data_prep.cnt, s=2, c="blue")

##

plt.subplot(2, 2, 3)

plt.title("Demand = f(Humidity)")

plt.scatter(x=data_prep.hum, y=data_prep.cnt, s=2, c="green")

##

plt.subplot(2, 2, 4)

plt.title("Demand = f(Wind speed)")

plt.scatter(x=data_prep.windspeed, y=data_prep.cnt, s=2, c="red")



plt.tight_layout()
# correlation degree of all the numerical features wrt to the total count of bike.

data_prep[["temp", "atemp", "hum", "windspeed", "cnt"]].corr()["cnt"].plot(kind="bar", title="Correlation of variable features wrt to total number of bikes")
# let's plot the evolution of total number of bikes wrt the different categorical features

cm = matplotlib.cm.get_cmap("rainbow")

fig, ax = plt.subplots(3, 3, figsize=(15, 15))

data_prep.groupby("season").mean()["cnt"].plot(ax=ax[0,0], kind="bar", color=cm(data_prep.groupby("season").mean()["cnt"]/np.max(data_prep.groupby("season").mean()["cnt"])))

data_prep.groupby("yr").mean()["cnt"].plot(ax=ax[0,1], kind="bar", color=cm(data_prep.groupby("yr").mean()["cnt"]/np.max(data_prep.groupby("yr").mean()["cnt"])))

data_prep.groupby("mnth").mean()["cnt"].plot(ax=ax[0,2], kind="bar", color=cm(data_prep.groupby("mnth").mean()["cnt"]/np.max(data_prep.groupby("mnth").mean()["cnt"])))

data_prep.groupby("hr").mean()["cnt"].plot(ax=ax[1,0], kind="bar", color=cm(data_prep.groupby("hr").mean()["cnt"]/np.max(data_prep.groupby("hr").mean()["cnt"])))

data_prep.groupby("holiday").mean()["cnt"].plot(ax=ax[1,1], kind="bar", color=cm(data_prep.groupby("holiday").mean()["cnt"]/np.max(data_prep.groupby("holiday").mean()["cnt"])))

data_prep.groupby("weekday").mean()["cnt"].plot(ax=ax[1,2], kind="bar", color=cm(data_prep.groupby("weekday").mean()["cnt"]/np.max(data_prep.groupby("weekday").mean()["cnt"])))

data_prep.groupby("workingday").mean()["cnt"].plot(ax=ax[2,0], kind="bar", color=cm(data_prep.groupby("workingday").mean()["cnt"]/np.max(data_prep.groupby("workingday").mean()["cnt"])))

data_prep.groupby("weathersit").mean()["cnt"].plot(ax=ax[2,1], kind="bar", color=cm(data_prep.groupby("weathersit").mean()["cnt"]/np.max(data_prep.groupby("weathersit").mean()["cnt"])))

plt.tight_layout()
# let's look at the hourly distribution

data_prep.groupby("hr").mean()["cnt"].plot(kind="bar", figsize=(16, 8), color=cm(data_prep.groupby("hr").mean()["cnt"]/np.max(data_prep.groupby("hr").mean()["cnt"])))
sns.boxplot(data=data_prep, x="cnt")
# another way to show this

data_prep.cnt.describe()
# check the boxplot in more details. print quartiles from 5% to 99% to check out outliers.

data_prep.quantile(np.append(np.arange(0.05, 0.96, 0.05), 0.99))["cnt"]
# let's check if numerical features are correlated with one another

sns.heatmap(data_prep[["temp", "atemp", "windspeed", "hum", "cnt"]].corr(), annot=True)
dropped = ["windspeed", "atemp", "workingday", "weekday", "yr"]

data_final = data_prep.drop(dropped, axis=1)
data_final.head()
# Let's check autocorrelation of cnt values

plt.acorr(data_final["cnt"].astype(float), maxlags=12)
df = np.log(data_final["cnt"])

df.hist(rwidth=0.9, bins=20, color="blue")
data_final["cnt"] = np.log(data_final["cnt"])
data_final.head()
# since cnt is correlated with itself, let's lag the cnt column and consider it as a feature

t1 = data_final["cnt"].shift(+1).to_frame()

t1.columns = ["t-1"]

t2 = data_final["cnt"].shift(+2).to_frame()

t2.columns = ["t-2"]

t3 = data_final["cnt"].shift(+3).to_frame()

t3.columns = ["t-3"]
data_lag = pd.concat([data_final, t1, t2, t3], axis=1)

data_lag.head()
# drop the NaN values

data_lag.dropna(inplace=True)
to_be_dummied = ["season", "mnth", "hr", "holiday", "weathersit"]

dummy_df = pd.get_dummies(data_lag[to_be_dummied].astype("category"), drop_first=True)

dummy_df.head()
# let's create ouf data finally pre-processed by concatenating the dummy variables with the numerical features.

dropped = ["season", "mnth", "holiday", "weathersit", "hr"]

df = pd.concat((data_lag.drop(dropped, axis=1), dummy_df), axis=1)

df.head()
from sklearn.model_selection import train_test_split
X = df.drop("cnt", axis=1)

y = df["cnt"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101) # test size of 25%
# we create the linear regression model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# let's fit it with the training set

lr.fit(X_train, y_train)
# check the score on the train set

lr.score(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
# this is the model predictions

test_pred = lr.predict(X_test)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

bins = None

sns.distplot(test_pred, ax=ax, color="blue", label="predictions", bins=bins)

sns.distplot(y_test, ax=ax, color="red", label="true", bins=bins)

ax.legend()
print(f"r2 score: {r2_score(y_test, test_pred):.2f}")

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.2f}")