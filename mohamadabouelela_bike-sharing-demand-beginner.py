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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/bike-datasets/train.csv", parse_dates = ["datetime"])

df.head()
df.shape
df.columns
df.info()
df.describe()
df.dtypes
df.isna().sum()
df = df.drop(["casual", "registered"], axis = 1)

df.head()
# add time paramteters to enrich datafarame



df["year"] = df.datetime.dt.year

df["month"] = df.datetime.dt.month

df["day"] = df.datetime.dt.day

df["hour"] = df.datetime.dt.hour

df.head()
# drop datetime column

df = df.drop(["datetime"], axis = 1)

df.head()
# Copy dataframe

df_new = df.copy()

df_new.head()
# visualize dataframe

df_new.hist(figsize = (20, 10));
fig = plt.figure(figsize = (20, 10))

fig.add_subplot(221, xlabel ="Temp").scatter(df_new["temp"], df_new["count"],c="red", s=25)

fig.add_subplot(222, xlabel ="Feels Like").scatter(df_new["atemp"], df_new["count"],c="orange", s=25)

fig.add_subplot(223, xlabel ="Humidity").scatter(df_new["humidity"], df_new["count"],c="green", s=25)

fig.add_subplot(224, xlabel ="Windspeed").scatter(df_new["windspeed"], df_new["count"],c="blue", s=25)

plt.tight_layout();
plt.scatter(df_new["hour"], df_new["count"], s =15, c="green");
plt.scatter(df_new["month"], df_new["count"], s =15, c="brown");
# split dataset

from sklearn.model_selection import train_test_split

np.random.seed(9)

X = df_new.drop("count", axis = 1)

y = df_new["count"]



X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Modelling using Random Forest regressor

from sklearn.ensemble import RandomForestRegressor

np.random.seed(9)

model = RandomForestRegressor()

model.fit(X_train, y_train)
y_preds = model.predict (X_test)
# Create function for evaluations

from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score 

def rmsle (y_test, y_preds):

    return np.sqrt(mean_squared_log_error(y_test, y_preds))



def model_score(model):

    scores = {"MAE" : mean_absolute_error(y_test, y_preds),

              "RMSLE" : rmsle(y_test, y_preds),

              "R2" : r2_score(y_test, y_preds)}

    eval = rmsle(y_test, y_preds)

    print (f"Model RMSLE = {eval:.3f}")

    return scores
model_score(model)
## Use test datset

df_test = pd.read_csv("../input/bike-datasets/test.csv", parse_dates=["datetime"])

df_test.head()
# test dataset preprocessing



df_test["year"] = df_test.datetime.dt.year

df_test["month"] = df_test.datetime.dt.month

df_test["day"] = df_test.datetime.dt.day

df_test["hour"] = df_test.datetime.dt.hour



df_test.head()
df_final = df_test.drop("datetime", axis = 1)

df_final.head()
y_preds_final = model.predict(df_final)
df_eval = pd.DataFrame()

df_eval["datetime"] = df_test["datetime"]

df_eval["count"] = y_preds_final
df_eval.head(10)
df_eval.to_csv("final.csv", sep = ",", index = False)