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
import os

import math

from pprint import pprint



import scipy as sp

import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt

import plotly



from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

from lightgbm import LGBMRegressor
train_filename = "train.csv"

test_filename = "test.csv"

df_train_raw = pd.read_csv(os.path.join(dirname, train_filename))

df_test_raw = pd.read_csv(os.path.join(dirname, test_filename))

df_train_raw.shape
df_train_raw.columns.values
df_train_clean = df_train_raw.drop(["Id"], axis=1)

df_test_clean = df_test_raw[~df_test_raw["Date"].isin(df_train_raw["Date"])]

df_test_clean = df_test_clean.drop(["ForecastId"], axis=1)

print("train shape: ", df_train_clean.shape)

print("test shape: ", df_test_clean.shape)
df_train_clean["Date"].unique()
df_test_clean["Date"].unique()
df_train_clean["ConfirmedCases"] = np.log1p(df_train_clean["ConfirmedCases"])

df_train_clean["Fatalities"] = np.log1p(df_train_clean["Fatalities"])

df_test_clean["ConfirmedCases"] = None

df_test_clean["Fatalities"] = None
df = pd.concat([df_train_clean, df_test_clean], sort=False).reset_index(drop=True)

print(df.shape)

print(df.columns.values)
df["Date"] = pd.to_datetime(df["Date"])
df["Province/State"].fillna(value = "NA", inplace = True)
## Aruba: 12.521110, -69.968338

df["Lat"].fillna(value = 12.521110, inplace = True)

df["Long"].fillna(value = -69.968338, inplace = True)
def get_trend(df, col):

  trend_col = "trend_{}".format(col)

  df[trend_col] = (df[col] - df[col].shift(fill_value=0)) / (df[col].shift(fill_value=0) + 0.0001)

  return df



def get_lagged_value(df, col, start, end):

  for lag in list(range(start, end)):

    lagged_col = "{}-{}D".format(col, lag)

    print(lagged_col)

    df[lagged_col] = df[col].shift(lag, fill_value=0)



  return df
df_lagged = df.copy(deep=True)

df_lagged = get_trend(df_lagged, "ConfirmedCases")

df_lagged.head()
df_lagged = get_trend(df_lagged, "Fatalities")

df_lagged.head()
df_lagged["C2F_ratio"] = df_lagged["ConfirmedCases"] / (df_lagged["Fatalities"] + 0.0001)

df_lagged.head()
df_lagged = get_lagged_value(df_lagged, "ConfirmedCases", 1, 7)

df_lagged = get_lagged_value(df_lagged, "Fatalities", 1, 7)
df_lagged = get_lagged_value(df_lagged, "trend_ConfirmedCases", 1, 7)

df_lagged = get_lagged_value(df_lagged, "trend_Fatalities", 1, 7)

df_lagged = get_lagged_value(df_lagged, "C2F_ratio", 1, 7)
base_date = pd.to_datetime("2020-01-01")

df_lagged.loc[:, "days_since"] = (df_lagged["Date"] - base_date).dt.days
df_lagged.loc[:,"Province/State"] = df_lagged["Province/State"].astype("category")

df_lagged.loc[:,"Country/Region"] = df_lagged["Country/Region"].astype("category")

# df_lagged.loc[:, "Date"] = df_lagged["Date"].astype("category")



# df_test.loc[:,"Province/State"] = df_test["Province/State"].astype("category")

# df_test.loc[:,"Country/Region"] = df_test["Country/Region"].astype("category")
df_lagged["ConfirmedCases"] = df_lagged["ConfirmedCases"].astype(float)

df_lagged["ConfirmedCases-1D"] = df_lagged["ConfirmedCases-1D"].astype(float)

df_lagged["ConfirmedCases-2D"] = df_lagged["ConfirmedCases-2D"].astype(float)

df_lagged["ConfirmedCases-3D"] = df_lagged["ConfirmedCases-3D"].astype(float)

df_lagged["ConfirmedCases-4D"] = df_lagged["ConfirmedCases-4D"].astype(float)

df_lagged["ConfirmedCases-5D"] = df_lagged["ConfirmedCases-5D"].astype(float)

df_lagged["ConfirmedCases-6D"] = df_lagged["ConfirmedCases-6D"].astype(float)

df_lagged["trend_ConfirmedCases"] = df_lagged["trend_ConfirmedCases"].astype(float)

df_lagged["trend_ConfirmedCases-1D"] = df_lagged["trend_ConfirmedCases-1D"].astype(float)

df_lagged["trend_ConfirmedCases-2D"] = df_lagged["trend_ConfirmedCases-2D"].astype(float)

df_lagged["trend_ConfirmedCases-3D"] = df_lagged["trend_ConfirmedCases-3D"].astype(float)

df_lagged["trend_ConfirmedCases-4D"] = df_lagged["trend_ConfirmedCases-4D"].astype(float)

df_lagged["trend_ConfirmedCases-5D"] = df_lagged["trend_ConfirmedCases-5D"].astype(float)

df_lagged["trend_ConfirmedCases-6D"] = df_lagged["trend_ConfirmedCases-6D"].astype(float)



df_lagged["Fatalities"] = df_lagged["Fatalities"].astype(float)

df_lagged["Fatalities-1D"] = df_lagged["Fatalities-1D"].astype(float)

df_lagged["Fatalities-2D"] = df_lagged["Fatalities-2D"].astype(float)

df_lagged["Fatalities-3D"] = df_lagged["Fatalities-3D"].astype(float)

df_lagged["Fatalities-4D"] = df_lagged["Fatalities-4D"].astype(float)

df_lagged["Fatalities-5D"] = df_lagged["Fatalities-5D"].astype(float)

df_lagged["Fatalities-6D"] = df_lagged["Fatalities-6D"].astype(float)

df_lagged["trend_Fatalities"] = df_lagged["trend_Fatalities"].astype(float)

df_lagged["trend_Fatalities-1D"] = df_lagged["trend_Fatalities-1D"].astype(float)

df_lagged["trend_Fatalities-2D"] = df_lagged["trend_Fatalities-2D"].astype(float)

df_lagged["trend_Fatalities-3D"] = df_lagged["trend_Fatalities-3D"].astype(float)

df_lagged["trend_Fatalities-4D"] = df_lagged["trend_Fatalities-4D"].astype(float)

df_lagged["trend_Fatalities-5D"] = df_lagged["trend_Fatalities-5D"].astype(float)

df_lagged["trend_Fatalities-6D"] = df_lagged["trend_Fatalities-6D"].astype(float)



df_lagged["C2F_ratio"] = df_lagged["C2F_ratio"].astype(float)

df_lagged["C2F_ratio-1D"] = df_lagged["C2F_ratio-1D"].astype(float)

df_lagged["C2F_ratio-2D"] = df_lagged["C2F_ratio-2D"].astype(float)

df_lagged["C2F_ratio-3D"] = df_lagged["C2F_ratio-3D"].astype(float)

df_lagged["C2F_ratio-4D"] = df_lagged["C2F_ratio-4D"].astype(float)

df_lagged["C2F_ratio-5D"] = df_lagged["C2F_ratio-5D"].astype(float)

df_lagged["C2F_ratio-6D"] = df_lagged["C2F_ratio-6D"].astype(float)
df_train = df_lagged[df_lagged["Date"] <= '2020-03-12']

df_valid = df_lagged[(df_lagged["Date"] > '2020-03-12') & (df_lagged["Date"] <= '2020-03-24')]

df_valid.head()
def rmsle(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    return np.sqrt(np.mean(np.power((y_pred - y_true), 2)))
target = "ConfirmedCases"

droppable = ["ConfirmedCases", "Fatalities", "trend_ConfirmedCases", "trend_Fatalities", "C2F_ratio"]

y_train = df_train[target]

X_train = df_train.drop(droppable, axis = 1)

y_valid = df_valid[target]

X_valid = df_valid.drop(droppable, axis = 1)
tscv = TimeSeriesSplit()

tscv
from hyperopt import hp, tpe

from hyperopt.fmin import fmin



from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer
mse_scorer = make_scorer(rmsle, greater_is_better=True, needs_proba=False)
X_train.drop(["Date"], axis=1, inplace=True)

X_valid.drop(["Date"], axis=1, inplace=True)
model_ConfirmedCases = LGBMRegressor(

        n_estimators=500,

        learning_rate=0.01,

        num_leaves=116,

        colsample_bytree=0.9997565232034884,

        subsample_for_bin=14000,

        reg_alpha=0.05603484531476548,

        reg_lambda=0.30358576246345437,

        min_child_samples=20,

        verbose=-1,

        n_jobs=-1,

        random_seed=42

        )



model_ConfirmedCases.fit(X_train, y_train)

# model_ConfirmedCases.booster_.save_model(os.path.join(dirname, "model_confirmed_cases.txt"))
y_pred = model_ConfirmedCases.predict(X_valid)

print("test score: {:.3f}".format(rmsle(y_valid, y_pred)))
df_allTrain = pd.concat([df_train, df_valid])

df_allTrain.columns.values
y = df_allTrain[target]

X = df_allTrain.drop([target, 'Fatalities', 'Date', 'trend_ConfirmedCases', 'trend_Fatalities', 'C2F_ratio'], axis=1)

model_ConfirmedCases.fit(X, y)

# model_ConfirmedCases.booster_.save_model(os.path.join(dirname, "model_confirmed_cases.txt"))
target_1 = "Fatalities"

y_train = df_train[target_1]

y_valid = df_valid[target_1]
model_Fatalities = LGBMRegressor(

        n_estimators=500,

        learning_rate=0.01,

        num_leaves=138,

        colsample_bytree=0.9811079071644806,

        subsample_for_bin=2000,

        reg_alpha=0.13184358714445044,

        reg_lambda=0.4247175016266793,

        min_child_samples=5,

        verbose=-1,

        n_jobs=-1,

        random_seed=42

        )



model_Fatalities.fit(X_train, y_train)

# model_Fatalities.booster_.save_model(os.path.join(dirname, "model_fatalities.txt"))
y_pred = model_ConfirmedCases.predict(X_valid)

print("test score: {:.3f}".format(rmsle(y_valid, y_pred)))
y = df_allTrain[target_1]

X = df_allTrain.drop([target, 'Fatalities', 'Date', 'trend_ConfirmedCases', 'trend_Fatalities', 'C2F_ratio'], axis=1)

model_Fatalities.fit(X, y)
def predict_slice(df, day):

  prev = day - 1

  print(day, " ", prev)



  df_ref = df[df["days_since"] == prev]



  df.loc[:, "ConfirmedCases-6D"] = df_ref["ConfirmedCases-5D"]

  df.loc[:, "ConfirmedCases-5D"] = df_ref["ConfirmedCases-4D"]

  df.loc[:, "ConfirmedCases-4D"] = df_ref["ConfirmedCases-3D"]

  df.loc[:, "ConfirmedCases-3D"] = df_ref["ConfirmedCases-2D"]

  df.loc[:, "ConfirmedCases-2D"] = df_ref["ConfirmedCases-1D"]

  df.loc[:, "ConfirmedCases-1D"] = df_ref["ConfirmedCases"]



  df.loc[:, "trend_ConfirmedCases-6D"] = df_ref["trend_ConfirmedCases-5D"]

  df.loc[:, "trend_ConfirmedCases-5D"] = df_ref["trend_ConfirmedCases-4D"]

  df.loc[:, "trend_ConfirmedCases-4D"] = df_ref["trend_ConfirmedCases-3D"]

  df.loc[:, "trend_ConfirmedCases-3D"] = df_ref["trend_ConfirmedCases-2D"]

  df.loc[:, "trend_ConfirmedCases-2D"] = df_ref["trend_ConfirmedCases-1D"]

  df.loc[:, "trend_ConfirmedCases-1D"] = (df_ref["ConfirmedCases-1D"] - df_ref["ConfirmedCases-2D"]) / (df_ref["ConfirmedCases-2D"] + 0.0001)



  df.loc[:, "Fatalities-6D"] = df_ref["Fatalities-5D"]

  df.loc[:, "Fatalities-5D"] = df_ref["Fatalities-4D"]

  df.loc[:, "Fatalities-4D"] = df_ref["Fatalities-3D"]

  df.loc[:, "Fatalities-3D"] = df_ref["Fatalities-2D"]

  df.loc[:, "Fatalities-2D"] = df_ref["Fatalities-1D"]

  df.loc[:, "Fatalities-1D"] = df_ref["Fatalities"]



  df.loc[:, "trend_Fatalities-6D"] = df_ref["trend_Fatalities-5D"]

  df.loc[:, "trend_Fatalities-5D"] = df_ref["trend_Fatalities-4D"]

  df.loc[:, "trend_Fatalities-4D"] = df_ref["trend_Fatalities-3D"]

  df.loc[:, "trend_Fatalities-3D"] = df_ref["trend_Fatalities-2D"]

  df.loc[:, "trend_Fatalities-2D"] = df_ref["trend_Fatalities-1D"]

  df.loc[:, "trend_Fatalities-1D"] = (df_ref["Fatalities-1D"] - df_ref["Fatalities-2D"]) / (df_ref["Fatalities-2D"] + 0.0001)



  df.loc[:, "C2F_ratio-6D"] = df_ref["C2F_ratio-5D"]

  df.loc[:, "C2F_ratio-5D"] = df_ref["C2F_ratio-4D"]

  df.loc[:, "C2F_ratio-4D"] = df_ref["C2F_ratio-3D"]

  df.loc[:, "C2F_ratio-3D"] = df_ref["C2F_ratio-2D"]

  df.loc[:, "C2F_ratio-2D"] = df_ref["C2F_ratio-1D"]

  df.loc[:, "C2F_ratio-1D"] = df_ref["ConfirmedCases-1D"] / (df_ref["Fatalities-1D"] + 0.0001)



  X = df.drop([target, target_1], axis = 1)

  df.loc[:, target] = model_ConfirmedCases.predict(X)

  df.loc[:, target_1] = model_Fatalities.predict(X)



  return df
cutoff = (pd.to_datetime("2020-03-12") - base_date).days

start = (pd.to_datetime("2020-03-24") - base_date).days

end = (pd.to_datetime("2020-04-23") - base_date).days

print(start)

print(end)

print(cutoff)
df_test = df_lagged[df_lagged["days_since"] >= cutoff].copy(deep=True)

df_test.drop([  'trend_ConfirmedCases'

              , 'trend_Fatalities'

              , 'C2F_ratio'

              , 'Date'], axis=1, inplace=True)
df_test.shape
day = cutoff



while day <= start:

  df_test.loc[df_test["days_since"] == day, target] = df_lagged.loc[df_lagged["days_since"] == day, target] 

  df_test.loc[df_test["days_since"] == day, target_1] = df_lagged.loc[df_lagged["days_since"] == day, target_1]  

  day = day + 1



while day <= end: 

  df_grp = df_test[df_test["days_since"] == day]

  df_test[df_test["days_since"] == day] = predict_slice(df_grp, day)

  day = day + 1
df_test["ConfirmedCases"] = np.expm1(df_test["ConfirmedCases"])

df_test["Fatalities"] = np.expm1(df_test["Fatalities"])
df_test = df_test[["days_since"

                  , "Province/State"

                  , "Country/Region"

                  , "Lat"

                  , "Long"

                  , "ConfirmedCases"

                  , "Fatalities"]]



df_test.head()
df_test.isna().sum()
df_test.shape
df_temp = df_test_raw.copy(deep=True)

df_temp["Province/State"].fillna(value = "NA", inplace=True)

## Aruba: 12.521110, -69.968338

df_temp["Lat"].fillna(value = 12.521110, inplace = True)

df_temp["Long"].fillna(value = -69.968338, inplace = True)



df_temp["Date"] = pd.to_datetime(df_temp["Date"])

df_temp["days_since"] = (df_temp["Date"] - base_date).dt.days

df_temp = pd.merge(df_temp, df_test, how="left", on=["days_since", "Province/State", "Country/Region"])

df_temp.shape
df_temp.isna().sum()
df_temp[df_temp[target].isna()]["Date"].unique()
df_temp.loc[df_temp[target] < 0, target] = 0

df_temp.loc[df_temp[target_1] < 0, target_1] = 0 
df_temp.loc[:, target] = df_temp[target].round(0).astype(int)

df_temp.loc[:, target_1] = df_temp[target_1].round(0).astype(int)
df_temp = df_temp[["ForecastId", target, target_1]]

df_temp.to_csv("submission.csv", index=False)